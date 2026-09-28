from __future__ import annotations
import math
from zoneinfo import ZoneInfo
from django.db.models import Q, Count
from django.shortcuts import get_object_or_404
from django.db import transaction
from django.utils import timezone as dj_tz, cache
from mobicrowd.models.ml_models import get_decoder
from mobicrowd.models.notifications import Notification
from mobicrowd.notify import notify_user, notify_users_bulk  # ← push + Notification creator
from mobicrowd.notify_tz import tz_map_for_users, iso_utc
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from PIL import Image
from typing import List, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs detected by TensorFlow:", gpus)  # should be []
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from django.conf import settings
from mobicrowd.models.Users import User
from mobicrowd.models.submisson import UserUploadLog, EventWorker
from celery import shared_task
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from transformers import CLIPProcessor, CLIPModel
import nltk
from nltk.corpus import stopwords
import sys
sys.path.append("mobicrowd/models/UNIQUE")
from mobicrowd.models.submisson import Submission, Photo, Event
from mobicrowd.views.apis.submission_flow.security import (
    FlowSecurityError,
    assert_event_open,
    assert_submission_owner,
    assert_worker_event_approved,
    sha256_directory,
    sha256_file,
)
import logging
import torch
import numpy as np
logger = logging.getLogger(__name__)
import os
from io import BytesIO
from pathlib import Path

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed

def save_to_cache(image_bytes):
    cache_dir = BASE_DIR / "tmp" / "reconstructed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(int(time.time() * 1000))
    file_path = cache_dir / f"{timestamp}.jpeg"

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return str(file_path)
autoencoder = get_decoder()
decoder = autoencoder  # adjust if you have separate decoder

RECON_VID_ROOT = BASE_DIR / "tmp" / "reconstructed videos"
class DecodeFromEmbeddingView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, submission_id, user_id=None):

        logger.info(
            (
                "[PHOTO-DECODE] request start "
                "submission_id=%s "
                "url_user_id=%s "
                "auth_user_id=%s "
                "request_keys=%s"
            ),
            submission_id,
            user_id,
            getattr(request.user, "id", None),
            list(request.data.keys()),
        )

        # =========================================================
        # 1. LEGACY URL ID MUST MATCH AUTHENTICATED USER
        # =========================================================

        if (
            user_id is not None
            and int(user_id) != int(request.user.id)
        ):
            logger.warning(
                (
                    "[PHOTO-DECODE] rejected: user mismatch "
                    "submission_id=%s "
                    "url_user_id=%s "
                    "auth_user_id=%s"
                ),
                submission_id,
                user_id,
                request.user.id,
            )

            return Response(
                {
                    "error": "Not allowed"
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # =========================================================
        # 2. EMBEDDING
        # =========================================================

        embedding = request.data.get("embedding")

        if not embedding:
            logger.warning(
                (
                    "[PHOTO-DECODE] rejected: "
                    "embedding missing "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "error": "Missing embedding"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            latent_flat = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(-1)

        except Exception:
            logger.exception(
                (
                    "[PHOTO-DECODE] invalid embedding "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "error":
                        "Invalid embedding."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        expected = 56 * 56 * 11

        logger.info(
            (
                "[PHOTO-DECODE] embedding received "
                "submission_id=%s "
                "embedding_size=%s "
                "expected_size=%s"
            ),
            submission_id,
            latent_flat.size,
            expected,
        )

        if latent_flat.size != expected:
            return Response(
                {
                    "error": (
                        f"Wrong embedding length "
                        f"{latent_flat.size}; "
                        f"expected {expected}."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        latent = latent_flat.reshape(
            (56, 56, 11)
        )

        latent_batch = np.expand_dims(
            latent,
            axis=0,
        )

        logger.info(
            (
                "[PHOTO-DECODE] latent prepared "
                "submission_id=%s "
                "latent_shape=%s "
                "batch_shape=%s"
            ),
            submission_id,
            latent.shape,
            latent_batch.shape,
        )

        try:

            # =====================================================
            # 3. LOAD + LOCK SUBMISSION
            # =====================================================

            with transaction.atomic():

                submission = (
                    Submission.objects
                    .select_for_update()
                    .select_related(
                        "worker",
                        "worker__user",
                        "event",
                        "photo",
                        "video",
                    )
                    .get(
                        pk=submission_id
                    )
                )

                logger.info(
                    (
                        "[PHOTO-DECODE] submission loaded "
                        "submission_id=%s "
                        "worker_id=%s "
                        "worker_user_id=%s "
                        "auth_user_id=%s "
                        "event_id=%s "
                        "photo_id=%s "
                        "video_id=%s "
                        "text_present=%s "
                        "status=%s "
                        "flow_stage=%s "
                        "flow_nonce=%s"
                    ),
                    submission.id,
                    submission.worker_id,
                    submission.worker.user_id,
                    request.user.id,
                    submission.event_id,
                    submission.photo_id,
                    submission.video_id,
                    bool(
                        (
                            submission.text
                            or ""
                        ).strip()
                    ),
                    submission.status,
                    submission.flow_stage,
                    submission.flow_nonce,
                )

                # =================================================
                # 4. SECURITY CHECKS
                # =================================================

                assert_submission_owner(
                    submission,
                    request.user,
                )

                assert_worker_event_approved(
                    submission=submission
                )

                assert_event_open(
                    submission
                )

                # =================================================
                # 5. MEDIA TYPE
                # =================================================

                if (
                    not submission.photo_id
                    or submission.video_id
                    or (
                        submission.text
                        or ""
                    ).strip()
                ):
                    logger.warning(
                        (
                            "[PHOTO-DECODE] rejected: "
                            "not photo submission "
                            "submission_id=%s "
                            "photo_id=%s "
                            "video_id=%s"
                        ),
                        submission_id,
                        submission.photo_id,
                        submission.video_id,
                    )

                    raise FlowSecurityError(
                        (
                            "This endpoint is only "
                            "for photo submissions."
                        )
                    )

                # =================================================
                # 6. STATUS
                # =================================================

                if (
                    submission.status
                    != Submission.PENDING
                ):
                    return Response(
                        {
                            "code":
                                "invalid_submission_sequence",
                            "current_stage":
                                submission.status,
                            "required_stage":
                                Submission.PENDING,
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                # =================================================
                # 7. FLOW STAGE
                # =================================================

                if (
                    submission.flow_stage
                    != Submission.FLOW_CREATED
                ):
                    return Response(
                        {
                            "code":
                                "invalid_submission_sequence",
                            "current_stage":
                                submission.flow_stage,
                            "required_stage":
                                Submission.FLOW_CREATED,
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                submission.flow_stage = (
                    Submission.FLOW_DECODING
                )

                submission.save(
                    update_fields=[
                        "flow_stage"
                    ]
                )

                # =================================================
                # 8. DECODE
                # =================================================

                logger.info(
                    (
                        "[PHOTO-DECODE] "
                        "decoder.predict start "
                        "submission_id=%s"
                    ),
                    submission_id,
                )

                decoded = decoder.predict(
                    latent_batch,
                    verbose=0,
                )[0]

                logger.info(
                    (
                        "[PHOTO-DECODE] "
                        "decoder.predict OK "
                        "submission_id=%s "
                        "decoded_shape=%s"
                    ),
                    submission_id,
                    getattr(
                        decoded,
                        "shape",
                        None,
                    ),
                )

                decoded_img = (
                    np.clip(
                        decoded,
                        0,
                        1,
                    )
                    * 255
                ).astype(
                    np.uint8
                )

                image = Image.fromarray(
                    decoded_img
                )

                buffer = BytesIO()

                image.save(
                    buffer,
                    format="JPEG",
                )

                image_bytes = (
                    buffer.getvalue()
                )

                # =================================================
                # 9. SAVE SERVER-SIDE RECONSTRUCTION
                # =================================================

                cache_dir = (
                    BASE_DIR
                    / "tmp"
                    / "reconstructed"
                )

                cache_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                saved_path = (
                    cache_dir
                    / (
                        f"{request.user.id}-"
                        f"{submission_id}-"
                        f"{submission.flow_nonce}.jpeg"
                    )
                ).resolve()

                saved_path.write_bytes(
                    image_bytes
                )

                # -------------------------------------------------
                # Compatibility path for existing frontend.
                #
                # This is what the old frontend expects from:
                #
                # decodeResponse.reconstructed_path ??
                # decodeResponse.path
                #
                # Phase A should still use the server-owned
                # submission.flow_artifact_path internally.
                # -------------------------------------------------

                relative_path = os.path.relpath(
                    saved_path,
                    BASE_DIR,
                )

                logger.info(
                    (
                        "[PHOTO-DECODE] artifact saved "
                        "submission_id=%s "
                        "absolute_path=%s "
                        "compatibility_path=%s"
                    ),
                    submission_id,
                    saved_path,
                    relative_path,
                )

                # =================================================
                # 10. DIGEST
                # =================================================

                digest = sha256_file(
                    saved_path
                )

                size_mb = (
                    len(image_bytes)
                    / (
                        1024
                        * 1024
                    )
                )

                # =================================================
                # 11. UPLOAD ACCOUNTING
                # =================================================

                log, _ = (
                    UserUploadLog.objects
                    .get_or_create(
                        submission=submission,
                        defaults={
                            "user":
                                request.user,
                            "size_mb":
                                0,
                        },
                    )
                )

                log.size_mb = (
                    (log.size_mb or 0)
                    + size_mb
                )

                log.user = request.user

                log.save(
                    update_fields=[
                        "user",
                        "size_mb",
                    ]
                )

                # =================================================
                # 12. SERVER-AUTHORITATIVE ARTIFACT
                # =================================================

                submission.flow_artifact_path = (
                    str(saved_path)
                )

                submission.flow_artifact_digest = (
                    digest
                )

                submission.flow_stage = (
                    Submission.FLOW_DECODED
                )

                submission.save(
                    update_fields=[
                        "flow_artifact_path",
                        "flow_artifact_digest",
                        "flow_stage",
                    ]
                )

                logger.info(
                    (
                        "[PHOTO-DECODE] SUCCESS "
                        "submission_id=%s "
                        "photo_id=%s "
                        "flow_stage=%s "
                        "artifact_path=%s"
                    ),
                    submission_id,
                    submission.photo_id,
                    submission.flow_stage,
                    submission.flow_artifact_path,
                )

            # =====================================================
            # 13. RESPONSE
            #
            # IMPORTANT:
            #
            # reconstructed_path + path are returned only for
            # backwards compatibility with the existing frontend.
            #
            # Server-side validation still relies on
            # submission.flow_artifact_path.
            # =====================================================

            return Response(
                {
                    "submission_id":
                        submission_id,

                    "flow_stage":
                        Submission.FLOW_DECODED,

                    "reconstructed_path":
                        relative_path,

                    "path":
                        relative_path,
                },
                status=status.HTTP_200_OK,
            )

        except Submission.DoesNotExist:

            logger.warning(
                (
                    "[PHOTO-DECODE] submission not found "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "error":
                        "Submission not found"
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        except FlowSecurityError as exc:

            logger.warning(
                (
                    "[PHOTO-DECODE] security rejection "
                    "submission_id=%s "
                    "reason=%s"
                ),
                submission_id,
                str(exc),
            )

            return Response(
                {
                    "error": str(exc)
                },
                status=status.HTTP_409_CONFLICT,
            )

        except Exception as exc:

            # =====================================================
            # RETRY-SAFE ROLLBACK
            # =====================================================

            Submission.objects.filter(
                pk=submission_id,
                status=Submission.PENDING,
                flow_stage=(
                    Submission.FLOW_DECODING
                ),
            ).update(
                flow_stage=(
                    Submission.FLOW_CREATED
                )
            )

            logger.exception(
                (
                    "[PHOTO-DECODE] failed "
                    "submission_id=%s "
                    "exception_type=%s "
                    "exception=%s"
                ),
                submission_id,
                type(exc).__name__,
                str(exc),
            )

            return Response(
                {
                    "error":
                        "Photo decode failed."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

def ensure_clean_decode_dir(user_id: int | str, submission_id: int | str) -> Path:
    """
    Ensure tmp/reconstructed videos/<user_id>-<submission_id>/ exists and is empty.
    Returns the directory Path.
    """
    run_dir = RECON_VID_ROOT / f"{user_id}-{submission_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous JPEGs so new run is deterministic
    for p in run_dir.glob("*.jpeg"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    for p in run_dir.glob("*.jpg"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    return run_dir

def save_numbered_jpegs_uint8_parallel(
        images_hw3_uint8: List, run_dir: Path, quality: int = 90, max_workers: int | None = None
) -> Tuple[float, List[str]]:
    """
    Save frames as 1.jpeg..N.jpeg in parallel (encode+write).
    Returns (total_mb, rel_paths) sorted by index.
    """
    # Default max_workers if not provided: based on CPU count
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4) * 2)  # Adjust based on CPU count

    def _encode_write(idx: int, arr) -> Tuple[int, float, str]:
        """
        Helper function to encode a single frame and write to disk.
        """
        img = Image.fromarray(arr)
        buf = BytesIO()
        # Note: optimize=False for faster encoding
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()

        dest = run_dir / f"{idx}.jpeg"
        dest.write_bytes(data)

        size_mb = len(data) / (1024 * 1024)  # Convert to MB
        rel = os.path.relpath(dest, BASE_DIR)
        return idx, size_mb, rel

    futures = []
    results = []

    # Parallel execution using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, arr in enumerate(images_hw3_uint8, start=1):
            futures.append(ex.submit(_encode_write, i, arr))

        # Collect results as they complete
        for fut in as_completed(futures):
            results.append(fut.result())

    # Sort results to maintain 1..N order
    results.sort(key=lambda x: x[0])

    # Calculate total MB and gather relative paths
    total_mb = sum(x[1] for x in results)
    rel_paths = [x[2] for x in results]

    return total_mb, rel_paths


LATENT_SHAPE = (56, 56, 11)
LATENT_SIZE = int(np.prod(LATENT_SHAPE))


def _save_frames(frames: np.ndarray, folder_name: str, folder_type: str) -> Path:
    save_root = Path(BASE_DIR) / "tmp" / "reconstructed videos"
    save_root.mkdir(parents=True, exist_ok=True)

    # Create the specific folder for this session using the folder name
    folder_path = save_root / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save each frame as a JPEG file
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img_path = folder_path / f"{i+1}.jpeg"
        img.save(img_path)

    # Return the folder path where the frames are stored
    return folder_path
class DecodeFromEmbeddingsBatchView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(
        self,
        request,
        submission_id,
        user_id=None,
    ):
        # =========================================================
        # LEGACY URL USER-ID COMPATIBILITY
        #
        # Authorization still comes from request.user.
        # =========================================================

        if (
            user_id is not None
            and int(user_id)
            != int(request.user.id)
        ):
            logger.warning(
                (
                    "[VIDEO-DECODE] rejected: "
                    "URL user mismatch "
                    "submission_id=%s "
                    "url_user_id=%s "
                    "auth_user_id=%s"
                ),
                submission_id,
                user_id,
                request.user.id,
            )

            return Response(
                {
                    "error": "Not allowed"
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        logger.info(
            (
                "[VIDEO-DECODE] request start "
                "submission_id=%s "
                "user_id=%s "
                "request_keys=%s"
            ),
            submission_id,
            request.user.id,
            list(request.data.keys()),
        )

        # =========================================================
        # EMBEDDINGS
        # =========================================================

        embeddings = request.data.get(
            "embeddings",
            [],
        )

        if (
            not isinstance(
                embeddings,
                list,
            )
            or not embeddings
        ):
            logger.warning(
                (
                    "[VIDEO-DECODE] rejected: "
                    "missing embeddings "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "error":
                        "Missing or empty "
                        "'embeddings' list"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(
            (
                "[VIDEO-DECODE] embeddings received "
                "submission_id=%s "
                "frame_count=%s"
            ),
            submission_id,
            len(embeddings),
        )

        # =========================================================
        # VALIDATE LATENT VECTORS
        # =========================================================

        latents = []

        for i, emb in enumerate(
            embeddings
        ):
            try:
                arr = np.asarray(
                    emb,
                    dtype=np.float32,
                ).reshape(-1)

            except Exception:
                logger.exception(
                    (
                        "[VIDEO-DECODE] "
                        "embedding conversion failed "
                        "submission_id=%s "
                        "frame_index=%s"
                    ),
                    submission_id,
                    i,
                )

                return Response(
                    {
                        "error":
                            f"Frame {i}: "
                            "invalid embedding."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if arr.size != LATENT_SIZE:
                logger.warning(
                    (
                        "[VIDEO-DECODE] rejected: "
                        "wrong embedding size "
                        "submission_id=%s "
                        "frame_index=%s "
                        "actual=%s "
                        "expected=%s"
                    ),
                    submission_id,
                    i,
                    arr.size,
                    LATENT_SIZE,
                )

                return Response(
                    {
                        "error": (
                            f"Frame {i}: "
                            f"wrong length "
                            f"{arr.size}, "
                            f"expected "
                            f"{LATENT_SIZE}"
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            latents.append(
                arr.reshape(
                    LATENT_SHAPE
                )
            )

        # =========================================================
        # REQUIRED VALIDATION FRAME COUNT
        # =========================================================

        expected_frames = int(
            getattr(
                settings,
                "SUBMISSION_VIDEO_VALIDATION_FRAMES",
                8,
            )
            or 8
        )

        if expected_frames != 8:
            logger.error(
                (
                    "[VIDEO-DECODE] invalid configuration "
                    "SUBMISSION_VIDEO_VALIDATION_FRAMES=%s"
                ),
                expected_frames,
            )

            # Current Qwen multi-frame relevance implementation
            # requires exactly eight frames.
            return Response(
                {
                    "error": (
                        "SUBMISSION_VIDEO_VALIDATION_FRAMES "
                        "must be 8 for the current validator."
                    )
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            # =====================================================
            # SERIALIZE WRITES PER SUBMISSION
            #
            # Prevents concurrent decode requests from racing frame
            # numbering or changing flow state incorrectly.
            # =====================================================

            with transaction.atomic():

                submission = (
                    Submission.objects
                    .select_for_update()
                    .select_related(
                        "worker",
                        "worker__user",
                        "event",
                        "video",
                        "photo",
                    )
                    .get(
                        pk=submission_id
                    )
                )

                logger.info(
                    (
                        "[VIDEO-DECODE] submission loaded "
                        "submission_id=%s "
                        "worker_id=%s "
                        "worker_user_id=%s "
                        "auth_user_id=%s "
                        "event_id=%s "
                        "video_id=%s "
                        "photo_id=%s "
                        "text_present=%s "
                        "status=%s "
                        "flow_stage=%s "
                        "flow_nonce=%s"
                    ),
                    submission.id,
                    submission.worker_id,
                    submission.worker.user_id,
                    request.user.id,
                    submission.event_id,
                    submission.video_id,
                    submission.photo_id,
                    bool(
                        (
                            submission.text
                            or ""
                        ).strip()
                    ),
                    submission.status,
                    submission.flow_stage,
                    submission.flow_nonce,
                )

                # =================================================
                # SECURITY
                # =================================================

                assert_submission_owner(
                    submission,
                    request.user,
                )

                assert_worker_event_approved(
                    submission=submission
                )

                assert_event_open(
                    submission
                )

                # =================================================
                # MUST BE A VIDEO SUBMISSION
                # =================================================

                if (
                    not submission.video_id
                    or submission.photo_id
                    or (
                        submission.text
                        or ""
                    ).strip()
                ):
                    logger.warning(
                        (
                            "[VIDEO-DECODE] rejected: "
                            "not a video submission "
                            "submission_id=%s "
                            "video_id=%s "
                            "photo_id=%s "
                            "text_present=%s"
                        ),
                        submission_id,
                        submission.video_id,
                        submission.photo_id,
                        bool(
                            (
                                submission.text
                                or ""
                            ).strip()
                        ),
                    )

                    raise FlowSecurityError(
                        (
                            "This endpoint is only "
                            "for video submissions."
                        )
                    )

                # =================================================
                # SUBMISSION STATUS
                # =================================================

                if (
                    submission.status
                    != Submission.PENDING
                ):
                    logger.warning(
                        (
                            "[VIDEO-DECODE] rejected: "
                            "invalid status "
                            "submission_id=%s "
                            "current=%s "
                            "required=%s"
                        ),
                        submission_id,
                        submission.status,
                        Submission.PENDING,
                    )

                    return Response(
                        {
                            "code":
                                "invalid_submission_sequence",
                            "current_stage":
                                submission.status,
                            "required_stage":
                                Submission.PENDING,
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                # =================================================
                # FLOW STAGE
                # =================================================

                if (
                    submission.flow_stage
                    not in (
                        Submission.FLOW_CREATED,
                        Submission.FLOW_DECODING,
                    )
                ):
                    logger.warning(
                        (
                            "[VIDEO-DECODE] rejected: "
                            "invalid flow stage "
                            "submission_id=%s "
                            "current=%s"
                        ),
                        submission_id,
                        submission.flow_stage,
                    )

                    return Response(
                        {
                            "code":
                                "invalid_submission_sequence",
                            "current_stage":
                                submission.flow_stage,
                            "required_stage": (
                                f"{Submission.FLOW_CREATED}"
                                f"|"
                                f"{Submission.FLOW_DECODING}"
                            ),
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                # =================================================
                # SERVER-OWNED RECONSTRUCTED DIRECTORY
                # =================================================

                run_dir = (
                    RECON_VID_ROOT
                    / (
                        f"{request.user.id}-"
                        f"{submission_id}-"
                        f"{submission.flow_nonce}"
                    )
                ).resolve()

                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # =================================================
                # INITIAL DECODE
                # =================================================

                if (
                    submission.flow_stage
                    == Submission.FLOW_CREATED
                ):
                    logger.info(
                        (
                            "[VIDEO-DECODE] "
                            "initializing decode directory "
                            "submission_id=%s "
                            "run_dir=%s"
                        ),
                        submission_id,
                        run_dir,
                    )

                    for p in run_dir.glob("*"):
                        if p.is_file():
                            p.unlink(
                                missing_ok=True
                            )

                    submission.flow_stage = (
                        Submission.FLOW_DECODING
                    )

                    submission.flow_artifact_path = (
                        str(run_dir)
                    )

                    submission.save(
                        update_fields=[
                            "flow_stage",
                            "flow_artifact_path",
                        ]
                    )

                # =================================================
                # EXISTING FRAMES
                # =================================================

                existing_files = sorted(
                    run_dir.glob("*.jpeg"),
                    key=lambda p:
                        int(p.stem),
                )

                existing = len(
                    existing_files
                )

                logger.info(
                    (
                        "[VIDEO-DECODE] frame state "
                        "submission_id=%s "
                        "existing=%s "
                        "incoming=%s "
                        "expected=%s"
                    ),
                    submission_id,
                    existing,
                    len(latents),
                    expected_frames,
                )

                if (
                    existing
                    + len(latents)
                    > expected_frames
                ):
                    logger.warning(
                        (
                            "[VIDEO-DECODE] rejected: "
                            "too many validation frames "
                            "submission_id=%s "
                            "existing=%s "
                            "incoming=%s "
                            "expected=%s"
                        ),
                        submission_id,
                        existing,
                        len(latents),
                        expected_frames,
                    )

                    return Response(
                        {
                            "error":
                                "Too many validation frames.",
                            "existing":
                                existing,
                            "incoming":
                                len(latents),
                            "expected_total":
                                expected_frames,
                        },
                        status=status.HTTP_409_CONFLICT,
                    )

                # =================================================
                # DECODE EMBEDDINGS
                # =================================================

                latent_batch = np.stack(
                    latents,
                    axis=0,
                )

                logger.info(
                    (
                        "[VIDEO-DECODE] decoder.predict start "
                        "submission_id=%s "
                        "batch_shape=%s"
                    ),
                    submission_id,
                    latent_batch.shape,
                )

                decoded = decoder.predict(
                    latent_batch,
                    batch_size=min(
                        4,
                        len(latents),
                    ),
                    verbose=0,
                )

                logger.info(
                    (
                        "[VIDEO-DECODE] decoder.predict OK "
                        "submission_id=%s "
                        "decoded_shape=%s"
                    ),
                    submission_id,
                    getattr(
                        decoded,
                        "shape",
                        None,
                    ),
                )

                decoded_uint8 = (
                    np.clip(
                        decoded,
                        0,
                        1,
                    )
                    * 255
                ).astype(
                    np.uint8
                )

                # =================================================
                # SAVE RECONSTRUCTED FRAMES
                # =================================================

                start_idx = (
                    existing
                    + 1
                )

                chunk_bytes = 0

                for (
                    offset,
                    frame,
                ) in enumerate(
                    decoded_uint8
                ):
                    idx = (
                        start_idx
                        + offset
                    )

                    target = (
                        run_dir
                        / f"{idx}.jpeg"
                    )

                    img = Image.fromarray(
                        frame
                    )

                    img.save(
                        target,
                        "JPEG",
                        quality=90,
                    )

                    chunk_bytes += (
                        target
                        .stat()
                        .st_size
                    )

                # =================================================
                # UPDATE ARTIFACT STATE
                # =================================================

                total_count = (
                    existing
                    + len(
                        decoded_uint8
                    )
                )

                digest = (
                    sha256_directory(
                        run_dir
                    )
                )

                # =================================================
                # UPLOAD ACCOUNTING
                # =================================================

                log, _ = (
                    UserUploadLog.objects
                    .get_or_create(
                        submission=submission,
                        defaults={
                            "user":
                                request.user,
                            "size_mb":
                                0,
                        },
                    )
                )

                log.user = (
                    request.user
                )

                log.size_mb = (
                    (log.size_mb or 0)
                    + chunk_bytes
                    / (
                        1024
                        * 1024
                    )
                )

                log.save(
                    update_fields=[
                        "user",
                        "size_mb",
                    ]
                )

                # =================================================
                # SERVER-AUTHORITATIVE ARTIFACT
                # =================================================

                submission.flow_artifact_path = (
                    str(run_dir)
                )

                submission.flow_artifact_digest = (
                    digest
                )

                submission.flow_stage = (
                    Submission.FLOW_DECODED
                    if (
                        total_count
                        == expected_frames
                    )
                    else
                    Submission.FLOW_DECODING
                )

                submission.save(
                    update_fields=[
                        "flow_artifact_path",
                        "flow_artifact_digest",
                        "flow_stage",
                    ]
                )

                logger.info(
                    (
                        "[VIDEO-DECODE] artifact saved "
                        "submission_id=%s "
                        "total_count=%s "
                        "flow_stage=%s "
                        "artifact_path=%s "
                        "digest=%s"
                    ),
                    submission_id,
                    total_count,
                    submission.flow_stage,
                    submission.flow_artifact_path,
                    submission.flow_artifact_digest,
                )

            # =====================================================
            # FRONTEND COMPATIBILITY PATH
            #
            # IMPORTANT:
            #
            # This value is returned because your CURRENT Angular
            # frontend checks:
            #
            #   reconstructed_path
            #   reconstructed_dir
            #   dir
            #   path
            #
            # before calling Phase A.
            #
            # The backend must NOT trust this value later.
            #
            # KickoffVideoSubmissionProcessingView already uses:
            #
            #   submission.flow_artifact_path
            #   submission.flow_artifact_digest
            #
            # which remain server-owned.
            # =====================================================

            try:
                compatibility_path = str(
                    run_dir.relative_to(
                        BASE_DIR
                    )
                )

            except ValueError:
                # Fallback only in case RECON_VID_ROOT
                # is outside BASE_DIR.
                compatibility_path = str(
                    run_dir
                )

            logger.info(
                (
                    "[VIDEO-DECODE] SUCCESS "
                    "submission_id=%s "
                    "saved=%s "
                    "start_index=%s "
                    "total_count=%s "
                    "expected_total=%s "
                    "flow_stage=%s "
                    "compatibility_path=%s"
                ),
                submission_id,
                len(decoded_uint8),
                start_idx,
                total_count,
                expected_frames,
                submission.flow_stage,
                compatibility_path,
            )

            # =====================================================
            # RESPONSE
            # =====================================================

            return Response(
                {
                    "submission_id":
                        submission_id,

                    "saved":
                        len(
                            decoded_uint8
                        ),

                    "start_index":
                        start_idx,

                    "total_count":
                        total_count,

                    "expected_total":
                        expected_frames,

                    "flow_stage":
                        submission.flow_stage,

                    # ---------------------------------------------
                    # Legacy/current frontend compatibility
                    # ---------------------------------------------

                    "reconstructed_path":
                        compatibility_path,

                    "reconstructed_dir":
                        compatibility_path,

                    "dir":
                        compatibility_path,

                    "path":
                        compatibility_path,
                },
                status=status.HTTP_200_OK,
            )

        # =========================================================
        # SUBMISSION DOES NOT EXIST
        # =========================================================

        except Submission.DoesNotExist:

            logger.warning(
                (
                    "[VIDEO-DECODE] "
                    "submission not found "
                    "submission_id=%s"
                ),
                submission_id,
            )

            return Response(
                {
                    "error":
                        "Submission not found"
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # =========================================================
        # SECURITY FAILURE
        # =========================================================

        except FlowSecurityError as exc:

            logger.warning(
                (
                    "[VIDEO-DECODE] "
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
                    "error":
                        str(exc)
                },
                status=status.HTTP_409_CONFLICT,
            )

        # =========================================================
        # TECHNICAL FAILURE
        # =========================================================

        except Exception as exc:

            logger.exception(
                (
                    "[VIDEO-DECODE] failed "
                    "submission_id=%s "
                    "exception_type=%s "
                    "exception=%s"
                ),
                submission_id,
                type(exc).__name__,
                str(exc),
            )

            return Response(
                {
                    "error":
                        "Video decode failed."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ─── CLIP setup for relevance ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

def clean_text(text: str) -> str:
    tokens = text.lower().split()
    return " ".join([t for t in tokens if t not in STOP_WORDS])



import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)



DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"



from datetime import timezone as dt_timezone
from django.utils import timezone

class ServerTimeView(APIView):
    def get(self, request):
        now_utc = timezone.now().astimezone(dt_timezone.utc)
        return Response({"utc_now": now_utc.isoformat()})


@shared_task
def send_broadcast(*, user_ids: list[int], title: str, body: str, payload: dict, priority: str = "high"):
    BATCH = 1000
    for i in range(0, len(user_ids), BATCH):
        chunk = user_ids[i:i+BATCH]
        for uid in chunk:
            notify_user(
                user_id=uid,
                event_type="system.maintenance",
                title=title,
                body=body,
                payload=payload,
                priority=priority,
            )

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
PROGRESS_KIND = "progress"                  # 80% of event duration elapsed
DEADLINE_KIND = "deadline"                  # T-30 minutes before deadline
INACTIVITY_KIND = "inactive_half"           # worker-specific participation midpoint
INACTIVITY_EVENT_TYPE = "event.reminder.inactive"  # Notification.event_type

END_KIND = "ended"                          # event deadline reached / event closed
END_EVENT_TYPE = "event.ended"              # Notification.event_type

# A worker approved with this amount of time or less remaining does not receive
# an inactivity reminder. Configure these values in Django settings if needed.
INACTIVITY_MIN_WINDOW_MINUTES = int(
    getattr(settings, "INACTIVITY_MIN_WINDOW_MINUTES", 60)
)
INACTIVITY_FINAL_QUIET_MINUTES = int(
    getattr(settings, "INACTIVITY_FINAL_QUIET_MINUTES", 30)
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _now():
    return dj_tz.now()


def _event_reminder_checkpoint(event: Event, kind: str):
    """Return the event-wide checkpoint for a progress or deadline reminder."""
    if not event.startdate or not event.deadline or event.startdate >= event.deadline:
        return None

    if kind == PROGRESS_KIND:
        total = event.deadline - event.startdate
        return event.startdate + timedelta(
            seconds=int(total.total_seconds() * 0.8)
        )

    if kind == DEADLINE_KIND:
        return event.deadline - timedelta(minutes=30)

    return None


def _worker_inactivity_timing(
    *,
    event_start,
    event_deadline,
    approved_at,
    joined_at,
):
    """
    Return (participation_start, reminder_at) for one approved worker.

    approved_at is authoritative. joined_at is only a compatibility fallback
    for approved rows created before the approved_at migration.
    """
    accepted_at = approved_at or joined_at
    if not event_start or not event_deadline or not accepted_at:
        return None
    if event_start >= event_deadline or accepted_at >= event_deadline:
        return None

    participation_start = max(event_start, accepted_at)
    participation_window = event_deadline - participation_start

    minimum_window = timedelta(
        minutes=max(0, INACTIVITY_MIN_WINDOW_MINUTES)
    )
    if participation_window <= minimum_window:
        return None

    reminder_at = participation_start + participation_window / 2
    final_quiet_start = event_deadline - timedelta(
        minutes=max(0, INACTIVITY_FINAL_QUIET_MINUTES)
    )

    # Close to the deadline, only the normal deadline reminder should be used.
    if reminder_at >= final_quiet_start:
        return None

    return participation_start, reminder_at

def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"

def _human_left_compact(mins: int) -> str:
    """Return compact time: '45m', '1h', '1h 15m'."""
    if mins < 60:
        return f"{mins}m"
    h, m = divmod(mins, 60)
    return f"{h}h {m}m" if m else f"{h}h"

def _human_left_phrase(mins: int) -> str:
    """Return natural phrase: 'less than a minute', '45 minutes', '1 hour 15 minutes'."""
    if mins <= 0:
        return "less than a minute"
    days, rem = divmod(mins, 1440)
    hours, minutes = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts) or "less than a minute"

def _progress_title(event_title: str, left_min: int, remaining: int) -> str:
    """Motivating title for the PROGRESS (80%) reminder."""
    left_phrase = _human_left_phrase(left_min)
    if remaining <= 2 and left_min <= 30:
        return f" {event_title}: last {remaining} shot{'s' if remaining != 1 else ''}!"
    if left_min <= 15:
        return f" {event_title}: almost there—{left_phrase} left"
    if left_min <= 60:
        return f"{event_title}: {left_phrase} remaining—keep going"
    return f" {event_title}: on track—{left_phrase} to go"

def _encouraging_title(kind: str, event_title: str, left_min: int) -> str:
    """Motivating title for the DEADLINE (T-30) reminder (fallback handles progress too)."""
    left_phrase = _human_left_phrase(left_min)
    if kind == DEADLINE_KIND:
        if left_min <= 5:   return f" {event_title}: last minutes—finish now"
        if left_min <= 15:  return f" {event_title}: only {left_phrase} left—don’t miss out"
        if left_min <= 30:  return f" {event_title}: {left_phrase} to wrap up"
        return f"{event_title}: ends in {left_phrase}"
    return f"⏳ {event_title}: {left_phrase} remaining"

def _already_sent(user_id: int, event_id: int, kind: str) -> bool:
    etype = "event.reminder.progress" if kind == PROGRESS_KIND else "event.reminder.deadline"
    return Notification.objects.filter(
        user_id=user_id,
        event_type=etype,
        payload__event_id=event_id,
    ).exists()

def _already_sent_inactive(user_id: int, event_id: int) -> bool:
    return Notification.objects.filter(
        user_id=user_id,
        event_type=INACTIVITY_EVENT_TYPE,
        payload__event_id=event_id,
        payload__kind=INACTIVITY_KIND,
    ).exists()

def _already_sent_ended(user_id: int, event_id: int) -> bool:
    return Notification.objects.filter(
        user_id=user_id,
        event_type=END_EVENT_TYPE,
        payload__event_id=event_id,
        payload__kind=END_KIND,
    ).exists()

def _event_end_any_sent(event_id: int) -> bool:
    """Event-level dedupe: if at least one 'ended' notification exists, assume this event was processed."""
    return Notification.objects.filter(
        event_type=END_EVENT_TYPE,
        payload__event_id=event_id,
        payload__kind=END_KIND,
    ).exists()


# ---------------------------------------------------------------------
# Time-based event reminders (80% + T-30)
# ---------------------------------------------------------------------
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def schedule_event_reminders(self, *, event_id: int) -> None:
    """
    Schedule reminders when an event is created/updated:
      - PROGRESS_KIND  at 80% of (start → deadline)
      - DEADLINE_KIND  at T-30 minutes
      - END_KIND       at deadline (event closed)

    IMPORTANT (Redis broker):
      - Do NOT schedule long ETA/countdown tasks beyond broker visibility_timeout.
        Otherwise Redis will re-deliver them and you will see duplicates.
      - We only schedule "near future" reminders (within visibility_timeout window).
      - Everything else is handled by periodic backfill tasks (run by Celery Beat).
    """
    try:
        event = Event.objects.only("id", "startdate", "deadline").get(id=event_id)
    except Event.DoesNotExist:
        return

    now = _now()
    start = event.startdate
    end = event.deadline
    if not start or not end or end <= now or start >= end:
        return

    # Redis safety window (seconds)
    vis = int(getattr(settings, "CELERY_BROKER_TRANSPORT_OPTIONS", {}).get("visibility_timeout", 600) or 600)
    max_safe = max(60, vis - 30)  # keep a buffer

    total = end - start
    t_progress = start + timedelta(seconds=int(total.total_seconds() * 0.8))   # 80%
    t_deadline = end - timedelta(minutes=30)                                  # T-30m
    t_end = end                                                               # at deadline

    def _schedule(task, kwargs: dict, when):
        if when <= now:
            # grace window: if it drifted into the past, send once immediately
            if (now - when) <= timedelta(minutes=5):
                task.delay(**kwargs)
            return

        countdown = max(1, math.ceil((when - now).total_seconds()))

        # Only schedule short countdowns; otherwise rely on backfill.
        if countdown <= max_safe:
            task.apply_async(kwargs=kwargs, countdown=countdown)

    _schedule(send_event_reminder, {"event_id": event.id, "kind": PROGRESS_KIND}, t_progress)
    _schedule(send_event_reminder, {"event_id": event.id, "kind": DEADLINE_KIND}, t_deadline)
    _schedule(send_event_ended_notification, {"event_id": event.id}, t_end)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def backfill_event_reminders(self) -> None:
    """
    Safety net: scan [now-12h, now+5m] for reminders that should fire and trigger them.
    Run this every ~5 minutes via Celery Beat.
    """
    now   = _now()
    past  = now - timedelta(hours=12)
    ahead = now + timedelta(minutes=5)

    qs = Event.objects.filter(deadline__gt=past).only("id", "startdate", "deadline")
    for e in qs:
        if not e.startdate or not e.deadline:
            continue

        total = e.deadline - e.startdate
        if total.total_seconds() <= 0:
            continue

        t_progress = e.startdate + timedelta(seconds=int(total.total_seconds() * 0.8))
        t_deadline = e.deadline - timedelta(minutes=30)

        if past <= t_progress <= ahead:
            send_event_reminder.delay(event_id=e.id, kind=PROGRESS_KIND)
        if past <= t_deadline <= ahead:
            send_event_reminder.delay(event_id=e.id, kind=DEADLINE_KIND)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def backfill_event_end_notifications(self) -> None:
    """
    Safety net: scan [now-12h, now+5m] for events whose deadline just passed and
    trigger the end-of-event notification.

    IMPORTANT:
      - Do NOT do event-level dedupe here (it can make some recipients miss notifications).
      - Per-user idempotency is enforced inside send_event_ended_notification.
    """
    now   = _now()
    past  = now - timedelta(hours=12)
    ahead = now + timedelta(minutes=5)

    qs = Event.objects.filter(deadline__gte=past, deadline__lte=ahead).only("id", "deadline")
    for e in qs:
        if not e.deadline:
            continue
        send_event_ended_notification.delay(event_id=e.id)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def send_event_reminder(self, *, event_id: int, kind: str) -> None:
    """
    Send a reminder to all APPROVED workers who still have attempts left.
    kind: 'progress' (80% elapsed) or 'deadline' (30 min left).

    Idempotency:
      - DB check via _already_sent(...)
      - Distributed lock via cache.add(...) to prevent races & duplicates
    """
    try:
        event = (Event.objects
                 .only("id", "title", "startdate", "deadline",
                       "max_photos_per_worker", "numberOfPhotos", "photo_reward" , "video_reward")
                 .get(id=event_id))
    except Event.DoesNotExist:
        return

    if kind not in {PROGRESS_KIND, DEADLINE_KIND}:
        return

    now = _now()
    if not event.startdate or not event.deadline:
        return
    if event.startdate >= event.deadline:
        return
    if event.deadline <= now:
        return

    checkpoint = _event_reminder_checkpoint(event, kind)
    if checkpoint is None:
        return

    # Backfill scans slightly ahead. Never send before the real checkpoint.
    if now < checkpoint:
        return

    # Shared timing for all recipients
    minutes_left = max(0, int((event.deadline - now).total_seconds() // 60))
    left_human_compact = _human_left_compact(minutes_left)

    # A worker receives an event-wide reminder only if the worker was approved
    # on or before that reminder's checkpoint.
    approved_before_checkpoint = (
        Q(approved_at__lte=checkpoint)
        | Q(approved_at__isnull=True, joined_at__lte=checkpoint)
    )

    # Eligible workers + used counts.
    rows = (
        EventWorker.objects
        .filter(
            event_id=event.id,
            status=EventWorker.APPROVED,
        )
        .filter(approved_before_checkpoint)
        .values(
            "worker_id",
            "worker__user_id",
            "approved_at",
            "joined_at",
        )
        .annotate(
            used_count=Count(
                "worker__submissions_list",
                filter=Q(
                    worker__submissions_list__event_id=event.id,
                    worker__submissions_list__status=Submission.APPROVED,
                ),
                distinct=True,
            )
        )
    )
    max_per = event.max_photos_per_worker or 0
    if max_per <= 0:
        return

    # Per-user tz (for localized tray time)
    user_ids = [r["worker__user_id"] for r in rows]
    tzmap = tz_map_for_users(user_ids)

    from django.core.cache import cache as dj_cache

    for row in rows:
        user_id = row["worker__user_id"]
        used = row["used_count"] or 0
        shots_left = max(0, max_per - used)
        if shots_left <= 0:
            continue

        # Title + CTA
        if kind == DEADLINE_KIND:
            etype = "event.reminder.deadline"
            base_title = _encouraging_title(kind, event.title, minutes_left)
            cta = "finish"
        else:
            etype = "event.reminder.progress"
            base_title = _progress_title(event.title, minutes_left, shots_left)
            cta = "continue"

        # Distributed lock to avoid races/duplicates (per user/event/kind)
        dedup_key = f"notif_once:{etype}:{user_id}:{event.id}:{kind}"
        if not dj_cache.add(dedup_key, "1", timeout=60 * 60 * 24 * 180):  # 180 days
            continue

        # DB idempotency (in case cache was flushed)
        if _already_sent(user_id, event.id, kind):
            continue

        # tz for this user (for friendly tray time)
        tzname = tzmap.get(user_id)
        try:
            tzinfo = ZoneInfo(tzname) if tzname else dj_tz.get_current_timezone()
        except Exception:
            tzinfo = dj_tz.get_current_timezone()

        end_local_dt = dj_tz.localtime(event.deadline, tzinfo)
        tz_label = end_local_dt.tzname()
        end_local_str = end_local_dt.strftime("%b %d, %H:%M")

        # UI body: short, upbeat, no dates (UI reads dates from payload)
        body_ui = f"{_plural(shots_left, 'photo')} left • Tap to {cta}"

        # Tray (push) body: compact + localized
        tray_bits = [
            f"{_plural(shots_left, 'photo')} left",
            f"Ends {end_local_str}" + (f" {tz_label}" if tz_label else "")
        ]
        tray_body = " • ".join(tray_bits)

        payload = {
            "type": "event",
            "event_id": event.id,
            "event_title": event.title,
            "kind": kind,                          # 'progress' | 'deadline'
            "remaining": shots_left,
            "max_per_worker": max_per,
            "photos_total": event.numberOfPhotos,
            "start_iso": iso_utc(event.startdate) if event.startdate else None,
            "deadline_iso": iso_utc(event.deadline),
            "tz_label": tz_label,
            "left_min": minutes_left,
            "left_human": left_human_compact,
            "cta": cta,
            "tray_body": tray_body,
        }

        try:
            notify_user(
                user_id=user_id,
                event_type=etype,
                title=base_title,
                body=body_ui,
                payload=payload,
                priority="high",
            )
        except Exception:
            # allow retry if send failed
            dj_cache.delete(dedup_key)
            raise

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def schedule_worker_inactivity_check(self, *, event_worker_id: int) -> None:
    """
    Schedule an inactivity check at the midpoint of this worker's own
    participation window.

    Call this right after a worker is APPROVED for an event.

    IMPORTANT (Redis broker):
      - Do NOT schedule long ETA/countdown tasks beyond broker visibility_timeout.
      - Only schedule near-future. Otherwise rely on Beat backfill.
    """
    try:
        ew = (EventWorker.objects
              .select_related("event")
              .get(id=event_worker_id))
    except EventWorker.DoesNotExist:
        return

    if ew.status != EventWorker.APPROVED:
        return

    event = ew.event
    now = _now()

    if not event.deadline or event.deadline <= now:
        return

    timing = _worker_inactivity_timing(
        event_start=event.startdate,
        event_deadline=event.deadline,
        approved_at=ew.approved_at,
        joined_at=ew.joined_at,
    )
    if timing is None:
        # The worker joined too late or the event dates are invalid.
        return

    _, reminder_at = timing
    final_quiet_start = event.deadline - timedelta(
        minutes=max(0, INACTIVITY_FINAL_QUIET_MINUTES)
    )

    if now >= final_quiet_start:
        return

    if reminder_at <= now:
        check_worker_inactivity.delay(event_worker_id=ew.id)
        return

    vis = int(getattr(settings, "CELERY_BROKER_TRANSPORT_OPTIONS", {}).get("visibility_timeout", 600) or 600)
    max_safe = max(60, vis - 30)

    countdown = max(1, math.ceil((reminder_at - now).total_seconds()))
    if countdown <= max_safe:
        check_worker_inactivity.apply_async(kwargs={"event_worker_id": ew.id}, countdown=countdown)
    # else: rely on backfill_worker_inactivity_checks (Beat)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def send_event_ended_notification(self, *, event_id: int) -> None:
    """
    End-of-event notification, sent once per recipient (workers + requester + admins).

    Idempotency:
      - DB check on Notification
      - cache-based distributed lock to prevent races when duplicate tasks run
    """
    try:
        event = Event.objects.select_related("requester", "requester__user").get(id=event_id)
    except Event.DoesNotExist:
        return

    now = _now()
    if not getattr(event, "deadline", None):
        return

    # old schedule fired but deadline extended
    if now + timedelta(seconds=30) < event.deadline:
        return

    worker_user_ids = list(
        EventWorker.objects
        .filter(event_id=event.id, status=EventWorker.APPROVED)
        .filter(
            Q(approved_at__lt=event.deadline)
            | Q(approved_at__isnull=True, joined_at__lt=event.deadline)
        )
        .values_list("worker__user_id", flat=True)
        .distinct()
    )

    requester_user_id = getattr(getattr(event, "requester", None), "user_id", None)

    admin_ids = set(
        User.objects.filter(role="Admin", is_active=True)
        .values_list("id", flat=True)
    )

    recipient_ids = set(worker_user_ids)
    if requester_user_id:
        recipient_ids.add(requester_user_id)
    recipient_ids.update(admin_ids)

    if not recipient_ids:
        return

    recipient_ids_list = list(recipient_ids)

    # DB idempotency
    already_sent_ids = set(
        Notification.objects.filter(
            user_id__in=recipient_ids_list,
            event_type=END_EVENT_TYPE,
            payload__event_id=event.id,
        ).values_list("user_id", flat=True)
    )

    from django.core.cache import cache as dj_cache

    locked_keys: list[str] = []
    to_send_ids: list[int] = []
    for uid in recipient_ids_list:
        if uid in already_sent_ids:
            continue
        dedup_key = f"notif_once:{END_EVENT_TYPE}:{uid}:{event.id}:{END_KIND}"
        if dj_cache.add(dedup_key, "1", timeout=60 * 60 * 24 * 365):  # 365 days
            locked_keys.append(dedup_key)
            to_send_ids.append(uid)

    if not to_send_ids:
        return

    tzmap = tz_map_for_users(to_send_ids)

    title = f"Event ended: {event.title}"
    items = []

    for uid in to_send_ids:
        tzname = tzmap.get(uid)
        try:
            tzinfo = ZoneInfo(tzname) if tzname else dj_tz.get_current_timezone()
        except Exception:
            tzinfo = dj_tz.get_current_timezone()

        end_local_dt = dj_tz.localtime(event.deadline, tzinfo)
        tz_label = end_local_dt.tzname()
        end_local_str = end_local_dt.strftime("%b %d, %H:%M")

        if uid in admin_ids:
            body_ui = "Event is now closed. Tap to open the admin dashboard."
            cta = "admin_dashboard"
        elif requester_user_id and uid == requester_user_id:
            body_ui = "Your event is now closed. Tap to review submissions."
            cta = "review"
        else:
            body_ui = "This event is now closed. Tap to view your submissions."
            cta = "results"

        tray_body = f"Ended {end_local_str}" + (f" {tz_label}" if tz_label else "")

        payload = {
            "type": "event",
            "event_id": event.id,
            "event_title": event.title,
            "kind": END_KIND,
            "deadline_iso": iso_utc(event.deadline),
            "ended_at_iso": iso_utc(event.deadline),
            "tz_label": tz_label,
            "tray_body": tray_body,
            "cta": cta,
        }

        items.append(
            {
                "user_id": uid,
                "title": title,
                "body": body_ui,
                "payload": payload,
                "priority": "high",
            }
        )

    try:
        # single DB write + WS + FCM fanout
        notify_users_bulk(
            event_type=END_EVENT_TYPE,
            items=items,
            match_filters={"payload__event_id": event.id},
        )
    except Exception:
        # allow retry if send failed
        for k in locked_keys:
            dj_cache.delete(k)
        raise

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def check_worker_inactivity(self, *, event_worker_id: int) -> None:
    """
    At the midpoint of the worker's own participation window, send one
    encouraging reminder if the worker has no submissions in any status.
    """
    try:
        ew = (EventWorker.objects
              .select_related("event", "worker")
              .get(id=event_worker_id))
    except EventWorker.DoesNotExist:
        return

    if ew.status != EventWorker.APPROVED:
        return

    event = ew.event
    now = _now()

    if not event.deadline or event.deadline <= now:
        return

    timing = _worker_inactivity_timing(
        event_start=event.startdate,
        event_deadline=event.deadline,
        approved_at=ew.approved_at,
        joined_at=ew.joined_at,
    )
    if timing is None:
        # The worker joined too late or the event dates are invalid.
        return

    participation_start, reminder_at = timing
    final_quiet_start = event.deadline - timedelta(
        minutes=max(0, INACTIVITY_FINAL_QUIET_MINUTES)
    )

    # A stale ETA task can fire early after an event deadline is extended.
    if now < reminder_at:
        schedule_worker_inactivity_check.delay(event_worker_id=ew.id)
        return

    # Do not send inactivity reminders near the deadline. The normal deadline
    # reminder handles that period.
    if now >= final_quiet_start:
        return

    # Inactive = no submissions of any status yet for this worker in this event
    has_any_submission = Submission.objects.filter(
        worker_id=ew.worker_id, event_id=event.id
    ).exists()
    if has_any_submission:
        return

    # Avoid duplicates
    user_id = ew.worker.user_id

    from django.core.cache import cache as dj_cache
    dedup_key = f"notif_once:{INACTIVITY_EVENT_TYPE}:{user_id}:{event.id}:{INACTIVITY_KIND}"
    if not dj_cache.add(dedup_key, "1", timeout=60 * 60 * 24 * 180):  # 180 days
        return

    if _already_sent_inactive(user_id, event.id):
        return

    # Per-user tz (for friendly tray time)
    tzmap = tz_map_for_users([user_id])
    tzname = tzmap.get(user_id)
    try:
        tzinfo = ZoneInfo(tzname) if tzname else dj_tz.get_current_timezone()
    except Exception:
        tzinfo = dj_tz.get_current_timezone()

    end_local_dt = dj_tz.localtime(event.deadline, tzinfo)
    tz_label = end_local_dt.tzname()
    end_local_str = end_local_dt.strftime("%b %d, %H:%M")

    max_per = event.max_photos_per_worker or 0
    # Title + bodies
    title = f"{event.title}: You have been missed — halfway already"
    body_ui = f"You can submit up to {_plural(max_per, 'photo')}. Tap to start"
    tray_body = f"No activity yet • Ends {end_local_str}" + (f" {tz_label}" if tz_label else "")

    payload = {
        "type": "event",
        "event_id": event.id,
        "event_title": event.title,
        "kind": INACTIVITY_KIND,
        "approved_at_iso": iso_utc(ew.approved_at or ew.joined_at),
        "participation_start_iso": iso_utc(participation_start),
        "half_iso": iso_utc(reminder_at),
        "deadline_iso": iso_utc(event.deadline),
        "tz_label": tz_label,
        "max_per_worker": max_per,
        "photos_total": event.numberOfPhotos,
        "tray_body": tray_body,
        "cta": "start",
    }

    try:
        notify_user(
            user_id=user_id,
            event_type=INACTIVITY_EVENT_TYPE,
            title=title,
            body=body_ui,     # UI list shows this; UI dates come from payload
            payload=payload,  # includes half/deadline ISO & tz
            priority="high",
        )
    except Exception:
        dj_cache.delete(dedup_key)
        raise

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def backfill_worker_inactivity_checks(self) -> None:
    """
    Safety net: for approved workers, run checks whose worker-specific
    participation midpoint occurred during the last 10 minutes.

    Notes:
    - The heavy per-worker logic runs inside check_worker_inactivity().
    - Long countdowns are intentionally left to this periodic task.
    """
    now = _now()
    past = now - timedelta(minutes=10)

    rows = (
        EventWorker.objects
        .filter(
            status=EventWorker.APPROVED,
            event__startdate__isnull=False,
            event__deadline__isnull=False,
            event__deadline__gt=now,
        )
        .values(
            "id",
            "approved_at",
            "joined_at",
            "event__startdate",
            "event__deadline",
        )
    )

    for row in rows.iterator(chunk_size=1000):
        timing = _worker_inactivity_timing(
            event_start=row["event__startdate"],
            event_deadline=row["event__deadline"],
            approved_at=row["approved_at"],
            joined_at=row["joined_at"],
        )
        if timing is None:
            continue

        _, reminder_at = timing
        if past <= reminder_at <= now:
            check_worker_inactivity.delay(event_worker_id=row["id"])

# --- Bootstrap reminders on Celery startup (ongoing events) ---
from celery.signals import worker_ready
from django.core.cache import cache as dj_cache  # IMPORTANT: this is the cache backend object

@shared_task
def bootstrap_ongoing_event_deadlines() -> None:
    """
    Runs once at worker startup.

    IMPORTANT:
      - Do NOT schedule long ETA/countdown tasks here (Redis redelivery storm).
      - Just run backfills once; Beat will keep doing it periodically.
    """
    backfill_event_reminders.delay()
    backfill_event_end_notifications.delay()
    backfill_worker_inactivity_checks.delay()