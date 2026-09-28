import secrets

from django.db import transaction
from django.utils import timezone
from django.utils.crypto import constant_time_compare

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework_simplejwt.tokens import RefreshToken

from mobicrowd.authentication.cookie_auth import (
    auth_success_response,
    coerce_bool,
)
from mobicrowd.authentication.session_security import (
    deactivate_other_sessions,
    hash_otp,
)
from mobicrowd.models.notifications import (
    DeviceToken,
    LoginTakeoverChallenge,
)


# =============================================================================
# Token issuance
# =============================================================================

def _issue_tokens(
    user,
    *,
    sid: str,
    kind: str,
    device_uid: str,
    remember_me: bool = False,
):
    """
    Create a session-bound JWT pair.

    auth_success_response() decides whether these tokens are:
      - returned in JSON for development/native clients, or
      - stored in HttpOnly cookies for production web.
    """
    refresh = RefreshToken.for_user(user)

    refresh["sid"] = sid
    refresh["kind"] = kind
    refresh["device_uid"] = device_uid
    refresh["remember"] = bool(remember_me)

    return {
        "id": user.id,
        "refresh": str(refresh),
        "access": str(refresh.access_token),
        "role": getattr(user, "role", ""),
    }


# =============================================================================
# Email-OTP takeover verification
# =============================================================================

class VerifyTakeoverOTPAPIView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        challenge_id = (
            request.data.get("challenge_id")
            or ""
        ).strip()

        otp = (
            request.data.get("otp")
            or ""
        ).strip()

        device_uid = (
            request.data.get("device_uid")
            or ""
        ).strip()

        platform = (
            request.data.get("platform")
            or ""
        ).strip().lower()

        remember_me = coerce_bool(
            request.data.get("remember_me")
        )

        # ---------------------------------------------------------------------
        # Request validation
        # ---------------------------------------------------------------------

        if (
            not challenge_id
            or not otp
            or not device_uid
            or platform
            not in (
                "android",
                "ios",
                "web",
            )
        ):
            return Response(
                {
                    "msg": (
                        "challenge_id, otp, "
                        "device_uid, and valid "
                        "platform are required"
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ---------------------------------------------------------------------
        # Verify challenge transactionally
        # ---------------------------------------------------------------------

        with transaction.atomic():
            challenge = (
                LoginTakeoverChallenge.objects
                .select_for_update()
                .select_related("user")
                .filter(
                    challenge_id=challenge_id,
                    method=(
                        LoginTakeoverChallenge
                        .METHOD_EMAIL_OTP
                    ),
                )
                .first()
            )

            if not challenge:
                return Response(
                    {
                        "msg": "Invalid challenge"
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if (
                challenge.status
                != LoginTakeoverChallenge
                .STATUS_PENDING
            ):
                return Response(
                    {
                        "msg": (
                            "Challenge is not pending"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if challenge.is_expired():
                challenge.status = (
                    LoginTakeoverChallenge
                    .STATUS_EXPIRED
                )

                challenge.decided_at = (
                    timezone.now()
                )

                challenge.save(
                    update_fields=[
                        "status",
                        "decided_at",
                    ]
                )

                return Response(
                    {
                        "msg": "Challenge expired"
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if (
                challenge.incoming_device_uid
                != device_uid
            ):
                return Response(
                    {
                        "msg": "Device mismatch"
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if (
                challenge.platform
                != platform
            ):
                return Response(
                    {
                        "msg": "Platform mismatch"
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            # -----------------------------------------------------------------
            # OTP brute-force limit
            # -----------------------------------------------------------------

            if challenge.otp_attempts >= 5:
                challenge.status = (
                    LoginTakeoverChallenge
                    .STATUS_DENIED
                )

                challenge.decided_at = (
                    timezone.now()
                )

                challenge.save(
                    update_fields=[
                        "status",
                        "decided_at",
                    ]
                )

                return Response(
                    {
                        "msg": "Too many attempts"
                    },
                    status=(
                        status
                        .HTTP_429_TOO_MANY_REQUESTS
                    ),
                )

            challenge.otp_attempts += 1

            expected_hash = (
                challenge.otp_hash
            )

            actual_hash = hash_otp(
                otp
            )

            if not constant_time_compare(
                expected_hash,
                actual_hash,
            ):
                challenge.save(
                    update_fields=[
                        "otp_attempts"
                    ]
                )

                return Response(
                    {
                        "msg": "Invalid OTP"
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            # -----------------------------------------------------------------
            # OTP verified
            # -----------------------------------------------------------------

            user = challenge.user
            kind = challenge.kind
            now = timezone.now()

            # Kill previous same-kind sessions before claiming the new session.
            deactivate_other_sessions(
                user,
                kind,
                keep_device_uid=device_uid,
            )

            sid = secrets.token_urlsafe(
                24
            )

            DeviceToken.objects.update_or_create(
                device_uid=device_uid,
                defaults={
                    "user": user,
                    "platform": platform,
                    "kind": kind,
                    "session_active": True,
                    "session_sid": sid,
                    "session_last_seen": now,
                    "is_active": True,
                    "last_seen": now,
                },
            )

            challenge.status = (
                LoginTakeoverChallenge
                .STATUS_USED
            )

            challenge.decided_at = now

            challenge.save(
                update_fields=[
                    "status",
                    "decided_at",
                    "otp_attempts",
                ]
            )

        # ---------------------------------------------------------------------
        # Issue replacement authentication
        # ---------------------------------------------------------------------

        tokens = _issue_tokens(
            user,
            sid=sid,
            kind=kind,
            device_uid=device_uid,
            remember_me=remember_me,
        )

        return auth_success_response(
            request,
            payload={
                "code": "LOGIN_SUCCESS",
                "message": "Login successful.",
                "msg": "Login Success",
            },
            tokens=tokens,
            platform=platform,
            remember_me=remember_me,
            status_code=status.HTTP_200_OK,
        )


# =============================================================================
# Existing authenticated session:
# list takeover requests
# =============================================================================

class PendingTakeoverChallengesAPIView(
    APIView
):
    permission_classes = [
        IsAuthenticated
    ]

    def get(self, request):
        token = request.auth

        kind = (
            token.get("kind")
            if token
            else None
        )

        if kind not in (
            "APP",
            "DASHBOARD",
        ):
            return Response(
                {
                    "msg": (
                        "Invalid session kind"
                    )
                },
                status=(
                    status
                    .HTTP_400_BAD_REQUEST
                ),
            )

        challenges = (
            LoginTakeoverChallenge.objects
            .filter(
                user=request.user,
                kind=kind,
                method=(
                    LoginTakeoverChallenge
                    .METHOD_ACTIVE_SESSION
                ),
                status=(
                    LoginTakeoverChallenge
                    .STATUS_PENDING
                ),
                expires_at__gt=(
                    timezone.now()
                ),
            )
            .order_by("-created_at")[:10]
        )

        data = [
            {
                "challenge_id": (
                    challenge.challenge_id
                ),
                "platform": (
                    challenge.platform
                ),
                "incoming_device_label": (
                    challenge
                    .incoming_device_label
                ),
                "created_at": (
                    challenge.created_at
                ),
                "expires_at": (
                    challenge.expires_at
                ),
            }
            for challenge in challenges
        ]

        return Response(
            data,
            status=status.HTTP_200_OK,
        )


# =============================================================================
# Existing authenticated session:
# approve or deny incoming takeover
# =============================================================================

class DecideTakeoverChallengeAPIView(
    APIView
):
    permission_classes = [
        IsAuthenticated
    ]

    def post(self, request):
        challenge_id = (
            request.data.get(
                "challenge_id"
            )
            or ""
        ).strip()

        decision = (
            request.data.get(
                "decision"
            )
            or ""
        ).strip().upper()

        if decision not in (
            "APPROVE",
            "DENY",
        ):
            return Response(
                {
                    "msg": (
                        "decision must be "
                        "APPROVE or DENY"
                    )
                },
                status=(
                    status
                    .HTTP_400_BAD_REQUEST
                ),
            )

        token = request.auth

        current_device_uid = (
            token.get("device_uid")
            if token
            else None
        )

        current_kind = (
            token.get("kind")
            if token
            else None
        )

        if (
            not current_device_uid
            or current_kind
            not in (
                "APP",
                "DASHBOARD",
            )
        ):
            return Response(
                {
                    "msg": (
                        "Invalid current session"
                    )
                },
                status=(
                    status
                    .HTTP_400_BAD_REQUEST
                ),
            )

        with transaction.atomic():
            challenge = (
                LoginTakeoverChallenge.objects
                .select_for_update()
                .filter(
                    challenge_id=(
                        challenge_id
                    ),
                    user=request.user,
                    kind=current_kind,
                    method=(
                        LoginTakeoverChallenge
                        .METHOD_ACTIVE_SESSION
                    ),
                )
                .first()
            )

            if not challenge:
                return Response(
                    {
                        "msg": (
                            "Challenge not found"
                        )
                    },
                    status=(
                        status
                        .HTTP_404_NOT_FOUND
                    ),
                )

            if (
                challenge.status
                != LoginTakeoverChallenge
                .STATUS_PENDING
            ):
                return Response(
                    {
                        "msg": (
                            "Challenge is not pending"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if challenge.is_expired():
                challenge.status = (
                    LoginTakeoverChallenge
                    .STATUS_EXPIRED
                )

                challenge.decided_at = (
                    timezone.now()
                )

                challenge.save(
                    update_fields=[
                        "status",
                        "decided_at",
                    ]
                )

                return Response(
                    {
                        "msg": (
                            "Challenge expired"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            # -----------------------------------------------------------------
            # Deny
            # -----------------------------------------------------------------

            if decision == "DENY":
                challenge.status = (
                    LoginTakeoverChallenge
                    .STATUS_DENIED
                )

                challenge.decided_at = (
                    timezone.now()
                )

                challenge.approved_by_device_uid = (
                    current_device_uid
                )

                challenge.save(
                    update_fields=[
                        "status",
                        "decided_at",
                        (
                            "approved_by_"
                            "device_uid"
                        ),
                    ]
                )

                return Response(
                    {
                        "msg": (
                            "Login request denied"
                        )
                    },
                    status=(
                        status.HTTP_200_OK
                    ),
                )

            # -----------------------------------------------------------------
            # Approve
            # -----------------------------------------------------------------

            challenge.status = (
                LoginTakeoverChallenge
                .STATUS_APPROVED
            )

            challenge.decided_at = (
                timezone.now()
            )

            challenge.approved_by_device_uid = (
                current_device_uid
            )

            challenge.save(
                update_fields=[
                    "status",
                    "decided_at",
                    (
                        "approved_by_"
                        "device_uid"
                    ),
                ]
            )

        return Response(
            {
                "msg": (
                    "Login request approved"
                )
            },
            status=status.HTTP_200_OK,
        )


# =============================================================================
# Incoming device:
# claim an already-approved takeover
# =============================================================================

class ClaimApprovedTakeoverAPIView(
    APIView
):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        challenge_id = (
            request.data.get(
                "challenge_id"
            )
            or ""
        ).strip()

        device_uid = (
            request.data.get(
                "device_uid"
            )
            or ""
        ).strip()

        platform = (
            request.data.get(
                "platform"
            )
            or ""
        ).strip().lower()

        remember_me = coerce_bool(
            request.data.get(
                "remember_me"
            )
        )

        if (
            not challenge_id
            or not device_uid
            or platform
            not in (
                "android",
                "ios",
                "web",
            )
        ):
            return Response(
                {
                    "msg": (
                        "challenge_id, "
                        "device_uid, and valid "
                        "platform are required"
                    )
                },
                status=(
                    status
                    .HTTP_400_BAD_REQUEST
                ),
            )

        with transaction.atomic():
            challenge = (
                LoginTakeoverChallenge.objects
                .select_for_update()
                .select_related("user")
                .filter(
                    challenge_id=(
                        challenge_id
                    ),
                    method=(
                        LoginTakeoverChallenge
                        .METHOD_ACTIVE_SESSION
                    ),
                )
                .first()
            )

            if not challenge:
                return Response(
                    {
                        "msg": (
                            "Challenge not found"
                        )
                    },
                    status=(
                        status
                        .HTTP_404_NOT_FOUND
                    ),
                )

            if (
                challenge.status
                != LoginTakeoverChallenge
                .STATUS_APPROVED
            ):
                return Response(
                    {
                        "msg": (
                            "Challenge is not approved"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if challenge.is_expired():
                challenge.status = (
                    LoginTakeoverChallenge
                    .STATUS_EXPIRED
                )

                challenge.decided_at = (
                    timezone.now()
                )

                challenge.save(
                    update_fields=[
                        "status",
                        "decided_at",
                    ]
                )

                return Response(
                    {
                        "msg": (
                            "Challenge expired"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if (
                challenge.incoming_device_uid
                != device_uid
            ):
                return Response(
                    {
                        "msg": (
                            "Device mismatch"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            if (
                challenge.platform
                != platform
            ):
                return Response(
                    {
                        "msg": (
                            "Platform mismatch"
                        )
                    },
                    status=(
                        status
                        .HTTP_400_BAD_REQUEST
                    ),
                )

            user = challenge.user
            kind = challenge.kind
            now = timezone.now()

            # Remove the previous active session(s).
            deactivate_other_sessions(
                user,
                kind,
                keep_device_uid=device_uid,
            )

            sid = secrets.token_urlsafe(
                24
            )

            DeviceToken.objects.update_or_create(
                device_uid=device_uid,
                defaults={
                    "user": user,
                    "platform": platform,
                    "kind": kind,
                    "session_active": True,
                    "session_sid": sid,
                    "session_last_seen": now,
                    "is_active": True,
                    "last_seen": now,
                },
            )

            challenge.status = (
                LoginTakeoverChallenge
                .STATUS_USED
            )

            challenge.decided_at = now

            challenge.save(
                update_fields=[
                    "status",
                    "decided_at",
                ]
            )

        # ---------------------------------------------------------------------
        # Issue authentication for newly approved device
        # ---------------------------------------------------------------------

        tokens = _issue_tokens(
            user,
            sid=sid,
            kind=kind,
            device_uid=device_uid,
            remember_me=remember_me,
        )

        return auth_success_response(
            request,
            payload={
                "code": "LOGIN_SUCCESS",
                "message": "Login successful.",
                "msg": "Login Success",
            },
            tokens=tokens,
            platform=platform,
            remember_me=remember_me,
            status_code=status.HTTP_200_OK,
        )