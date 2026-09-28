# mobicrowd/views/apis/submission_flow/views.py
from __future__ import annotations

from django.core import signing
from django.db import transaction
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from celery.exceptions import TimeoutError as CeleryTimeoutError

from mobicrowd.models.submisson import Event, Submission
from mobicrowd.views.apis.submission_flow.image_tasks import (
    process_original_and_finalize,
    stage_relevance,
    stage_redundancy, logger,
)
from mobicrowd.views.apis.submission_flow.video_tasks import (
    process_original_video_and_finalize,
    stage_relevance_video,
    stage_redundancy_video,
)
from mobicrowd.views.apis.submission_flow.security import (
    FlowSecurityError,
    assert_event_open,
    assert_finalize_token_matches,
    assert_submission_owner,
    assert_worker_event_approved,
    approved_capacity_available,
    issue_finalize_token,
    load_finalize_token,
    sanitize_filename,
    sha256_directory,
    sha256_file,
    submission_media_type,
)
from django.conf import settings

def _sequence_error(*, current: str, required: str) -> Response:
    return Response(
        {
            "ok": False,
            "code": "invalid_submission_sequence",
            "current_stage": current,
            "required_stage": required,
            "message": f"Submission is in '{current}' stage; '{required}' is required.",
        },
        status=status.HTTP_409_CONFLICT,
    )


def _rollback_validation(submission_id: int) -> None:
    # Only infrastructure failures are retryable. Content rejection remains final.
    Submission.objects.filter(
        pk=submission_id,
        status=Submission.PENDING,
        flow_stage=Submission.FLOW_VALIDATING,
    ).update(
        flow_stage=Submission.FLOW_CREATED,
        flow_artifact_path=None,
        flow_artifact_digest=None,
    )


def _sync_refused_stage(submission_id: int) -> None:
    Submission.objects.filter(
        pk=submission_id,
        status=Submission.REFUSED,
    ).update(flow_stage=Submission.FLOW_REFUSED)


def _load_owned_submission_for_update(*, submission_id: int, request_user, media_type: str):
    submission = (
        Submission.objects.select_for_update()
        .select_related("worker", "worker__user", "event", "photo", "video")
        .get(pk=submission_id)
    )
    assert_submission_owner(submission, request_user)
    actual = submission_media_type(submission)
    if actual != media_type:
        raise FlowSecurityError(f"This endpoint is for {media_type} submissions, not {actual}.")
    assert_worker_event_approved(submission=submission)
    assert_event_open(submission)
    return submission


class KickoffSubmissionProcessingView(APIView):

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, submission_id: int):
        try:
            with transaction.atomic():
                submission = _load_owned_submission_for_update(
                    submission_id=submission_id,
                    request_user=request.user,
                    media_type="photo",
                )

                if submission.status != Submission.PENDING:
                    return _sequence_error(current=submission.status, required=Submission.PENDING)
                if submission.flow_stage != Submission.FLOW_DECODED:
                    return _sequence_error(
                        current=submission.flow_stage,
                        required=Submission.FLOW_DECODED,
                    )
                if not submission.flow_artifact_path or not submission.flow_artifact_digest:
                    return Response(
                        {"ok": False, "code": "missing_decoded_artifact"},
                        status=status.HTTP_409_CONFLICT,
                    )

                current_digest = sha256_file(submission.flow_artifact_path)
                if current_digest != submission.flow_artifact_digest:
                    return Response(
                        {"ok": False, "code": "decoded_artifact_tampered"},
                        status=status.HTTP_409_CONFLICT,
                    )

                artifact_path = submission.flow_artifact_path
                description = submission.event.keywords or ""
                submission.flow_stage = Submission.FLOW_VALIDATING
                submission.save(update_fields=["flow_stage"])

        except Submission.DoesNotExist:
            return Response({"error": "Submission not found"}, status=status.HTTP_404_NOT_FOUND)
        except FlowSecurityError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_409_CONFLICT)

        try:
            rel = stage_relevance.apply_async(
                kwargs={
                    "submission_id": int(submission_id),
                    "user_id": int(request.user.id),
                    "reconstructed_path": artifact_path,
                    "description": description,
                }
            ).get(timeout=400, propagate=True)
        except CeleryTimeoutError:
            _rollback_validation(submission_id)
            return Response(
                {"ok": False, "stage": "relevance", "message": "Relevance timed out."},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except Exception:
            _rollback_validation(submission_id)
            return Response(
                {"ok": False, "stage": "relevance", "message": "Relevance validation failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if rel.get("stop"):
            _sync_refused_stage(submission_id)
            return Response(
                {
                    "ok": False,
                    "decision": "REJECT",
                    "stage": "relevance",
                    "submission_id": int(submission_id),
                    "analysis": rel.get("analysis", {}),
                },
                status=status.HTTP_200_OK,
            )

        try:
            red = stage_redundancy.apply_async(args=[rel]).get(timeout=30, propagate=True)
        except CeleryTimeoutError:
            _rollback_validation(submission_id)
            return Response(
                {"ok": False, "stage": "redundancy", "message": "Redundancy timed out."},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except Exception:
            _rollback_validation(submission_id)
            return Response(
                {"ok": False, "stage": "redundancy", "message": "Redundancy validation failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if red.get("stop"):
            _sync_refused_stage(submission_id)
            return Response(
                {
                    "ok": False,
                    "decision": "REJECT",
                    "stage": "redundancy",
                    "submission_id": int(submission_id),
                    "analysis": red.get("analysis", red),
                },
                status=status.HTTP_200_OK,
            )

        try:
            with transaction.atomic():
                submission = _load_owned_submission_for_update(
                    submission_id=submission_id,
                    request_user=request.user,
                    media_type="photo",
                )
                if submission.status == Submission.REFUSED:
                    submission.flow_stage = Submission.FLOW_REFUSED
                    submission.save(update_fields=["flow_stage"])
                    return Response({"ok": False, "decision": "REJECT"}, status=status.HTTP_200_OK)
                if submission.flow_stage != Submission.FLOW_VALIDATING:
                    return _sequence_error(
                        current=submission.flow_stage,
                        required=Submission.FLOW_VALIDATING,
                    )

                submission.flow_stage = Submission.FLOW_VALIDATED
                submission.flow_validated_at = timezone.now()
                # Relevance task removes the temporary artifact after extracting its embedding.
                submission.flow_artifact_path = None
                submission.save(
                    update_fields=["flow_stage", "flow_validated_at", "flow_artifact_path"]
                )
                finalize_token = issue_finalize_token(submission)
        except FlowSecurityError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_409_CONFLICT)

        return Response(
            {
                "ok": True,
                "decision": "PASS_PHASE_A",
                "submission_id": int(submission_id),
                "finalize_token": finalize_token,
                "analysis": {
                    "relevance": rel.get("analysis", {}),
                    "redundancy": red.get("analysis", {}),
                },
            },
            status=status.HTTP_200_OK,
        )

class UploadOriginalAndContinueView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, submission_id: int):
        token = str(
            request.data.get("finalize_token")
            or request.data.get("upload_token")
            or ""
        ).strip()

        base64_image = request.data.get("base64Image")
        file_name = request.data.get("fileName")

        logger.info(
            (
                "[PHOTO-ORIGINAL] request "
                "submission_id=%s "
                "user_id=%s "
                "token_present=%s "
                "base64_present=%s "
                "file_name=%r "
                "request_keys=%s"
            ),
            submission_id,
            getattr(request.user, "id", None),
            bool(token),
            bool(base64_image),
            file_name,
            list(request.data.keys()),
        )

        # =========================================================
        # ORIGINAL MEDIA MUST EXIST
        # =========================================================

        if not base64_image or not file_name:
            logger.warning(
                (
                    "[PHOTO-ORIGINAL] rejected: "
                    "missing original media "
                    "submission_id=%s "
                    "base64_present=%s "
                    "file_name_present=%s"
                ),
                submission_id,
                bool(base64_image),
                bool(file_name),
            )

            return Response(
                {
                    "code": "original_media_missing",
                    "error": (
                        "base64Image and fileName "
                        "are required"
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # =========================================================
        # FILE NAME
        # =========================================================

        try:
            clean_name = sanitize_filename(
                file_name
            )

        except FlowSecurityError as exc:
            logger.warning(
                (
                    "[PHOTO-ORIGINAL] invalid filename "
                    "submission_id=%s "
                    "reason=%s"
                ),
                submission_id,
                str(exc),
            )

            return Response(
                {
                    "code": "invalid_filename",
                    "error": str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # =========================================================
        # OPTIONAL TOKEN
        #
        # New clients:
        #   finalize_token is verified normally.
        #
        # Existing frontend:
        #   token is absent.
        #   Server-side FLOW_VALIDATED + authenticated ownership
        #   becomes the compatibility proof.
        # =========================================================

        token_payload = None

        if token:
            try:
                token_payload = load_finalize_token(
                    token
                )

            except signing.SignatureExpired:
                logger.warning(
                    (
                        "[PHOTO-ORIGINAL] "
                        "finalize token expired "
                        "submission_id=%s"
                    ),
                    submission_id,
                )

                return Response(
                    {
                        "code":
                            "finalize_token_expired"
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            except signing.BadSignature:
                logger.warning(
                    (
                        "[PHOTO-ORIGINAL] "
                        "invalid finalize token "
                        "submission_id=%s"
                    ),
                    submission_id,
                )

                return Response(
                    {
                        "code":
                            "invalid_finalize_token"
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            except FlowSecurityError as exc:
                logger.warning(
                    (
                        "[PHOTO-ORIGINAL] "
                        "token security failure "
                        "submission_id=%s "
                        "reason=%s"
                    ),
                    submission_id,
                    str(exc),
                )

                return Response(
                    {
                        "code":
                            "invalid_finalize_token",
                        "message":
                            str(exc),
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

        # =========================================================
        # LOAD / LOCK SUBMISSION
        # =========================================================

        try:
            with transaction.atomic():

                submission = (
                    _load_owned_submission_for_update(
                        submission_id=submission_id,
                        request_user=request.user,
                        media_type="photo",
                    )
                )

                logger.info(
                    (
                        "[PHOTO-ORIGINAL] submission loaded "
                        "submission_id=%s "
                        "worker_id=%s "
                        "event_id=%s "
                        "photo_id=%s "
                        "status=%s "
                        "flow_stage=%s "
                        "validated_at=%s "
                        "token_present=%s"
                    ),
                    submission.id,
                    submission.worker_id,
                    submission.event_id,
                    submission.photo_id,
                    submission.status,
                    submission.flow_stage,
                    submission.flow_validated_at,
                    bool(token),
                )

                # =================================================
                # MUST STILL BE PENDING
                # =================================================

                if (
                    submission.status
                    != Submission.PENDING
                ):
                    logger.warning(
                        (
                            "[PHOTO-ORIGINAL] rejected: "
                            "wrong status "
                            "submission_id=%s "
                            "current=%s "
                            "required=%s"
                        ),
                        submission_id,
                        submission.status,
                        Submission.PENDING,
                    )

                    return _sequence_error(
                        current=submission.status,
                        required=Submission.PENDING,
                    )

                # =================================================
                # PHASE A MUST HAVE PASSED
                # =================================================

                if (
                    submission.flow_stage
                    != Submission.FLOW_VALIDATED
                ):
                    logger.warning(
                        (
                            "[PHOTO-ORIGINAL] rejected: "
                            "wrong flow stage "
                            "submission_id=%s "
                            "current=%s "
                            "required=%s"
                        ),
                        submission_id,
                        submission.flow_stage,
                        Submission.FLOW_VALIDATED,
                    )

                    return _sequence_error(
                        current=submission.flow_stage,
                        required=(
                            Submission.FLOW_VALIDATED
                        ),
                    )

                # =================================================
                # NEW-CLIENT TOKEN VALIDATION
                # =================================================

                if token_payload is not None:
                    assert_finalize_token_matches(
                        payload=token_payload,
                        submission=submission,
                        request_user=request.user,
                        expected_media_type="photo",
                    )

                    logger.info(
                        (
                            "[PHOTO-ORIGINAL] "
                            "finalize token verified "
                            "submission_id=%s"
                        ),
                        submission_id,
                    )

                # =================================================
                # OLD-FRONTEND COMPATIBILITY
                #
                # No token was supplied. Require a recent,
                # server-generated successful Phase-A state.
                # =================================================

                else:
                    if not submission.flow_validated_at:
                        logger.warning(
                            (
                                "[PHOTO-ORIGINAL] rejected: "
                                "FLOW_VALIDATED without timestamp "
                                "submission_id=%s"
                            ),
                            submission_id,
                        )

                        return Response(
                            {
                                "code":
                                    "validation_timestamp_missing",
                                "message": (
                                    "Submission validation "
                                    "state is incomplete."
                                ),
                            },
                            status=status.HTTP_409_CONFLICT,
                        )

                    ttl_seconds = int(
                        getattr(
                            settings,
                            (
                                "SUBMISSION_FINALIZE_"
                                "TOKEN_TTL_SECONDS"
                            ),
                            900,
                        )
                    )

                    validation_age_seconds = (
                        timezone.now()
                        - submission.flow_validated_at
                    ).total_seconds()

                    if (
                        validation_age_seconds
                        > ttl_seconds
                    ):
                        logger.warning(
                            (
                                "[PHOTO-ORIGINAL] rejected: "
                                "validation expired "
                                "submission_id=%s "
                                "age=%.3f "
                                "ttl=%s"
                            ),
                            submission_id,
                            validation_age_seconds,
                            ttl_seconds,
                        )

                        return Response(
                            {
                                "code":
                                    "validated_state_expired",
                                "message": (
                                    "The validated upload "
                                    "window has expired."
                                ),
                            },
                            status=status.HTTP_403_FORBIDDEN,
                        )

                    logger.info(
                        (
                            "[PHOTO-ORIGINAL] "
                            "legacy frontend accepted "
                            "without finalize token "
                            "submission_id=%s "
                            "validation_age=%.3fs"
                        ),
                        submission_id,
                        validation_age_seconds,
                    )

                # =================================================
                # CAPACITY CHECK
                # =================================================

                submission.event = (
                    Event.objects
                    .select_for_update()
                    .get(
                        pk=submission.event_id
                    )
                )

                (
                    capacity_ok,
                    capacity_reason,
                ) = approved_capacity_available(
                    submission
                )

                if not capacity_ok:
                    logger.warning(
                        (
                            "[PHOTO-ORIGINAL] rejected: "
                            "capacity reached "
                            "submission_id=%s "
                            "reason=%s"
                        ),
                        submission_id,
                        capacity_reason,
                    )

                    return Response(
                        {
                            "code":
                                capacity_reason,
                            "message": (
                                "Submission capacity "
                                "has been reached."
                            ),
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                # =================================================
                # CLAIM FINALIZATION
                # =================================================

                flow_nonce = str(
                    submission.flow_nonce
                )

                submission.flow_stage = (
                    Submission.FLOW_FINALIZING
                )

                submission.save(
                    update_fields=[
                        "flow_stage"
                    ]
                )

                logger.info(
                    (
                        "[PHOTO-ORIGINAL] "
                        "state -> FINALIZING "
                        "submission_id=%s "
                        "flow_nonce=%s"
                    ),
                    submission_id,
                    flow_nonce,
                )

        except Submission.DoesNotExist:
            logger.warning(
                (
                    "[PHOTO-ORIGINAL] "
                    "submission not found "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "code":
                        "submission_not_found",
                    "error":
                        "Submission not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        except FlowSecurityError as exc:
            logger.warning(
                (
                    "[PHOTO-ORIGINAL] "
                    "security rejection "
                    "submission_id=%s "
                    "reason=%s"
                ),
                submission_id,
                str(exc),
            )

            return Response(
                {
                    "code":
                        "original_upload_not_allowed",
                    "message":
                        str(exc),
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # =========================================================
        # QUEUE PHASE B
        # =========================================================

        try:
            logger.info(
                (
                    "[PHOTO-ORIGINAL] "
                    "queueing Phase B "
                    "submission_id=%s "
                    "file_name=%s"
                ),
                submission_id,
                clean_name,
            )

            process_original_and_finalize.delay(
                submission_id=int(
                    submission_id
                ),
                user_id=int(
                    request.user.id
                ),
                base64_image=base64_image,
                file_name=clean_name,
                flow_nonce=flow_nonce,
            )

        except Exception:
            logger.exception(
                (
                    "[PHOTO-ORIGINAL] "
                    "Phase B queue failed "
                    "submission_id=%s"
                ),
                submission_id,
            )

            # Allow retry if Celery queueing itself failed.
            Submission.objects.filter(
                pk=submission_id,
                status=Submission.PENDING,
                flow_stage=(
                    Submission.FLOW_FINALIZING
                ),
                flow_nonce=flow_nonce,
            ).update(
                flow_stage=(
                    Submission.FLOW_VALIDATED
                )
            )

            return Response(
                {
                    "code":
                        "finalization_queue_failed"
                },
                status=(
                    status
                    .HTTP_503_SERVICE_UNAVAILABLE
                ),
            )

        # =========================================================
        # SUCCESS
        # =========================================================

        logger.info(
            (
                "[PHOTO-ORIGINAL] SUCCESS "
                "submission_id=%s "
                "flow_stage=%s"
            ),
            submission_id,
            Submission.FLOW_FINALIZING,
        )

        return Response(
            {
                "ok": True,
                "queued": True,
                "submission_id":
                    submission_id,
                "flow_stage":
                    Submission.FLOW_FINALIZING,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class KickoffVideoSubmissionProcessingView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, submission_id: int):
        try:
            with transaction.atomic():
                submission = _load_owned_submission_for_update(
                    submission_id=submission_id,
                    request_user=request.user,
                    media_type="video",
                )
                if submission.status != Submission.PENDING:
                    return _sequence_error(current=submission.status, required=Submission.PENDING)
                if submission.flow_stage != Submission.FLOW_DECODED:
                    return _sequence_error(current=submission.flow_stage, required=Submission.FLOW_DECODED)
                if not submission.flow_artifact_path or not submission.flow_artifact_digest:
                    return Response({"code": "missing_decoded_artifact"}, status=status.HTTP_409_CONFLICT)

                current_digest = sha256_directory(submission.flow_artifact_path)
                if current_digest != submission.flow_artifact_digest:
                    return Response({"code": "decoded_artifact_tampered"}, status=status.HTTP_409_CONFLICT)

                frames_dir = submission.flow_artifact_path
                description = submission.event.keywords or ""
                submission.flow_stage = Submission.FLOW_VALIDATING
                submission.save(update_fields=["flow_stage"])
        except Submission.DoesNotExist:
            return Response({"error": "Submission not found"}, status=status.HTTP_404_NOT_FOUND)
        except FlowSecurityError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_409_CONFLICT)

        try:
            rel = stage_relevance_video.apply_async(
                kwargs={
                    "submission_id": int(submission_id),
                    "user_id": int(request.user.id),
                    "frames_dir": frames_dir,
                    "description": description,
                }
            ).get(timeout=500, propagate=True)
        except CeleryTimeoutError:
            _rollback_validation(submission_id)
            return Response({"ok": False, "stage": "relevance", "message": "Video relevance timed out."}, status=status.HTTP_504_GATEWAY_TIMEOUT)
        except Exception:
            _rollback_validation(submission_id)
            return Response({"ok": False, "stage": "relevance", "message": "Video relevance validation failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if rel.get("stop"):
            _sync_refused_stage(submission_id)
            return Response({"ok": False, "decision": "REJECT", "stage": "relevance", "submission_id": submission_id, "analysis": rel.get("analysis", {})})

        try:
            red = stage_redundancy_video.apply_async(args=[rel]).get(timeout=60, propagate=True)
        except CeleryTimeoutError:
            _rollback_validation(submission_id)
            return Response({"ok": False, "stage": "redundancy", "message": "Video redundancy timed out."}, status=status.HTTP_504_GATEWAY_TIMEOUT)
        except Exception:
            _rollback_validation(submission_id)
            return Response({"ok": False, "stage": "redundancy", "message": "Video redundancy validation failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if red.get("stop"):
            _sync_refused_stage(submission_id)
            return Response({"ok": False, "decision": "REJECT", "stage": "redundancy", "submission_id": submission_id, "analysis": red.get("analysis", red)})

        try:
            with transaction.atomic():
                submission = _load_owned_submission_for_update(
                    submission_id=submission_id,
                    request_user=request.user,
                    media_type="video",
                )
                if submission.status == Submission.REFUSED:
                    submission.flow_stage = Submission.FLOW_REFUSED
                    submission.save(update_fields=["flow_stage"])
                    return Response({"ok": False, "decision": "REJECT"})
                if submission.flow_stage != Submission.FLOW_VALIDATING:
                    return _sequence_error(current=submission.flow_stage, required=Submission.FLOW_VALIDATING)

                submission.flow_stage = Submission.FLOW_VALIDATED
                submission.flow_validated_at = timezone.now()
                submission.flow_artifact_path = None
                submission.save(update_fields=["flow_stage", "flow_validated_at", "flow_artifact_path"])
                finalize_token = issue_finalize_token(submission)
        except FlowSecurityError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_409_CONFLICT)

        return Response(
            {
                "ok": True,
                "decision": "PASS_PHASE_A",
                "submission_id": submission_id,
                "finalize_token": finalize_token,
                "analysis": {"relevance": rel.get("analysis", {}), "redundancy": red.get("analysis", {})},
            },
            status=status.HTTP_200_OK,
        )


class UploadOriginalVideoAndContinueView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, submission_id: int):

        token = str(
            request.data.get("finalize_token")
            or request.data.get("upload_token")
            or ""
        ).strip()

        base64_video = request.data.get("base64Video")
        file_name = request.data.get("fileName")

        logger.info(
            (
                "[VIDEO-ORIGINAL] request "
                "submission_id=%s "
                "user_id=%s "
                "token_present=%s "
                "video_present=%s "
                "file_name=%r "
                "request_keys=%s"
            ),
            submission_id,
            getattr(request.user, "id", None),
            bool(token),
            bool(base64_video),
            file_name,
            list(request.data.keys()),
        )

        # =========================================================
        # ORIGINAL VIDEO IS REQUIRED
        # =========================================================

        if not base64_video or not file_name:
            logger.warning(
                (
                    "[VIDEO-ORIGINAL] rejected: "
                    "missing video/file "
                    "submission_id=%s "
                    "video_present=%s "
                    "file_name_present=%s"
                ),
                submission_id,
                bool(base64_video),
                bool(file_name),
            )

            return Response(
                {
                    "code": "original_video_missing",
                    "error": (
                        "base64Video and fileName "
                        "are required"
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # =========================================================
        # SANITIZE FILE NAME
        # =========================================================

        try:
            clean_name = sanitize_filename(file_name)

        except FlowSecurityError as exc:
            logger.warning(
                (
                    "[VIDEO-ORIGINAL] invalid filename "
                    "submission_id=%s "
                    "reason=%s"
                ),
                submission_id,
                str(exc),
            )

            return Response(
                {
                    "code": "invalid_filename",
                    "error": str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # =========================================================
        # OPTIONAL FINALIZE TOKEN
        #
        # New clients can still send the token.
        # Existing frontend sends no token.
        # =========================================================

        token_payload = None

        if token:
            try:
                token_payload = load_finalize_token(token)

            except signing.SignatureExpired:
                logger.warning(
                    (
                        "[VIDEO-ORIGINAL] token expired "
                        "submission_id=%s"
                    ),
                    submission_id,
                )

                return Response(
                    {
                        "code": "finalize_token_expired"
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            except signing.BadSignature:
                logger.warning(
                    (
                        "[VIDEO-ORIGINAL] invalid token "
                        "submission_id=%s"
                    ),
                    submission_id,
                )

                return Response(
                    {
                        "code": "invalid_finalize_token"
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            except FlowSecurityError as exc:
                logger.warning(
                    (
                        "[VIDEO-ORIGINAL] "
                        "token validation failed "
                        "submission_id=%s "
                        "reason=%s"
                    ),
                    submission_id,
                    str(exc),
                )

                return Response(
                    {
                        "code": "invalid_finalize_token",
                        "message": str(exc),
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

        # =========================================================
        # LOAD AND LOCK SUBMISSION
        # =========================================================

        try:
            with transaction.atomic():

                submission = (
                    _load_owned_submission_for_update(
                        submission_id=submission_id,
                        request_user=request.user,
                        media_type="video",
                    )
                )

                logger.info(
                    (
                        "[VIDEO-ORIGINAL] submission loaded "
                        "submission_id=%s "
                        "worker_id=%s "
                        "event_id=%s "
                        "video_id=%s "
                        "status=%s "
                        "flow_stage=%s "
                        "validated_at=%s "
                        "token_present=%s"
                    ),
                    submission.id,
                    submission.worker_id,
                    submission.event_id,
                    submission.video_id,
                    submission.status,
                    submission.flow_stage,
                    submission.flow_validated_at,
                    bool(token),
                )

                # =================================================
                # MUST STILL BE PENDING
                # =================================================

                if submission.status != Submission.PENDING:
                    logger.warning(
                        (
                            "[VIDEO-ORIGINAL] rejected: "
                            "wrong status "
                            "submission_id=%s "
                            "current=%s "
                            "required=%s"
                        ),
                        submission_id,
                        submission.status,
                        Submission.PENDING,
                    )

                    return _sequence_error(
                        current=submission.status,
                        required=Submission.PENDING,
                    )

                # =================================================
                # PHASE A MUST HAVE PASSED
                # =================================================

                if (
                    submission.flow_stage
                    != Submission.FLOW_VALIDATED
                ):
                    logger.warning(
                        (
                            "[VIDEO-ORIGINAL] rejected: "
                            "wrong flow stage "
                            "submission_id=%s "
                            "current=%s "
                            "required=%s"
                        ),
                        submission_id,
                        submission.flow_stage,
                        Submission.FLOW_VALIDATED,
                    )

                    return _sequence_error(
                        current=submission.flow_stage,
                        required=Submission.FLOW_VALIDATED,
                    )

                # =================================================
                # TOKEN-SUPPLIED CLIENT
                # =================================================

                if token_payload is not None:

                    assert_finalize_token_matches(
                        payload=token_payload,
                        submission=submission,
                        request_user=request.user,
                        expected_media_type="video",
                    )

                    logger.info(
                        (
                            "[VIDEO-ORIGINAL] "
                            "finalize token verified "
                            "submission_id=%s"
                        ),
                        submission_id,
                    )

                # =================================================
                # CURRENT FRONTEND — NO TOKEN
                #
                # Require a fresh server-created FLOW_VALIDATED
                # state instead.
                # =================================================

                else:

                    if not submission.flow_validated_at:
                        logger.warning(
                            (
                                "[VIDEO-ORIGINAL] rejected: "
                                "missing validation timestamp "
                                "submission_id=%s"
                            ),
                            submission_id,
                        )

                        return Response(
                            {
                                "code":
                                    "validation_timestamp_missing",
                                "message": (
                                    "Submission validation "
                                    "state is incomplete."
                                ),
                            },
                            status=status.HTTP_409_CONFLICT,
                        )

                    ttl_seconds = int(
                        getattr(
                            settings,
                            "SUBMISSION_FINALIZE_TOKEN_TTL_SECONDS",
                            900,
                        )
                    )

                    validation_age_seconds = (
                        timezone.now()
                        - submission.flow_validated_at
                    ).total_seconds()

                    if validation_age_seconds > ttl_seconds:
                        logger.warning(
                            (
                                "[VIDEO-ORIGINAL] rejected: "
                                "validated state expired "
                                "submission_id=%s "
                                "age_seconds=%.3f "
                                "ttl_seconds=%s"
                            ),
                            submission_id,
                            validation_age_seconds,
                            ttl_seconds,
                        )

                        return Response(
                            {
                                "code":
                                    "validated_state_expired",
                                "message": (
                                    "The validated upload "
                                    "window has expired."
                                ),
                            },
                            status=status.HTTP_403_FORBIDDEN,
                        )

                    logger.info(
                        (
                            "[VIDEO-ORIGINAL] "
                            "legacy frontend accepted "
                            "without finalize token "
                            "submission_id=%s "
                            "validation_age=%.3fs"
                        ),
                        submission_id,
                        validation_age_seconds,
                    )

                # =================================================
                # LOCK EVENT AND CHECK CAPACITY
                # =================================================

                submission.event = (
                    Event.objects
                    .select_for_update()
                    .get(pk=submission.event_id)
                )

                capacity_ok, capacity_reason = (
                    approved_capacity_available(
                        submission
                    )
                )

                if not capacity_ok:
                    logger.warning(
                        (
                            "[VIDEO-ORIGINAL] rejected: "
                            "capacity reached "
                            "submission_id=%s "
                            "reason=%s"
                        ),
                        submission_id,
                        capacity_reason,
                    )

                    return Response(
                        {
                            "code": capacity_reason,
                            "message": (
                                "Submission capacity "
                                "has been reached."
                            ),
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                # =================================================
                # CLAIM FINALIZATION
                #
                # Once changed from VALIDATED -> FINALIZING,
                # the same upload cannot simply be replayed.
                # =================================================

                flow_nonce = str(
                    submission.flow_nonce
                )

                submission.flow_stage = (
                    Submission.FLOW_FINALIZING
                )

                submission.save(
                    update_fields=[
                        "flow_stage"
                    ]
                )

                logger.info(
                    (
                        "[VIDEO-ORIGINAL] "
                        "state -> FINALIZING "
                        "submission_id=%s "
                        "flow_nonce=%s"
                    ),
                    submission_id,
                    flow_nonce,
                )

        except Submission.DoesNotExist:
            logger.warning(
                (
                    "[VIDEO-ORIGINAL] "
                    "submission not found "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "code": "submission_not_found",
                    "error": "Submission not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        except FlowSecurityError as exc:
            logger.warning(
                (
                    "[VIDEO-ORIGINAL] "
                    "security rejection "
                    "submission_id=%s "
                    "user_id=%s "
                    "reason=%s"
                ),
                submission_id,
                request.user.id,
                str(exc),
            )

            return Response(
                {
                    "code":
                        "original_video_upload_not_allowed",
                    "message":
                        str(exc),
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # =========================================================
        # QUEUE PHASE B
        # =========================================================

        try:
            logger.info(
                (
                    "[VIDEO-ORIGINAL] "
                    "queueing Phase B "
                    "submission_id=%s "
                    "user_id=%s "
                    "file_name=%s"
                ),
                submission_id,
                request.user.id,
                clean_name,
            )

            process_original_video_and_finalize.delay(
                submission_id=int(submission_id),
                user_id=int(request.user.id),
                base64_video=base64_video,
                file_name=clean_name,
                flow_nonce=flow_nonce,
            )

        except Exception:
            logger.exception(
                (
                    "[VIDEO-ORIGINAL] "
                    "Phase B queue failed "
                    "submission_id=%s"
                ),
                submission_id,
            )

            # Queueing failed, so restore VALIDATED and
            # allow the same client to retry.
            Submission.objects.filter(
                pk=submission_id,
                status=Submission.PENDING,
                flow_stage=(
                    Submission.FLOW_FINALIZING
                ),
                flow_nonce=flow_nonce,
            ).update(
                flow_stage=Submission.FLOW_VALIDATED
            )

            return Response(
                {
                    "code":
                        "finalization_queue_failed"
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # =========================================================
        # SUCCESS
        # =========================================================

        logger.info(
            (
                "[VIDEO-ORIGINAL] SUCCESS "
                "submission_id=%s "
                "flow_stage=%s"
            ),
            submission_id,
            Submission.FLOW_FINALIZING,
        )

        return Response(
            {
                "ok": True,
                "queued": True,
                "submission_id": submission_id,
                "flow_stage":
                    Submission.FLOW_FINALIZING,
            },
            status=status.HTTP_202_ACCEPTED,
        )