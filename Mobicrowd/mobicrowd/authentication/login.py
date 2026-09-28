import base64
import json
import logging
import os
import secrets

import requests

from django.conf import settings
from django.contrib.auth import authenticate
from django.utils import timezone

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework_simplejwt.tokens import RefreshToken

from mobicrowd.authentication.cookie_auth import (
    auth_success_response,
    coerce_bool,
)
from mobicrowd.authentication.email_sending import (
    send_login_takeover_otp_email,
)
from mobicrowd.authentication.session_security import (
    create_active_session_challenge,
    create_email_otp_challenge,
    deactivate_stale_sessions,
)
from mobicrowd.models.Users import Requester, User
from mobicrowd.models.notifications import DeviceToken


logger = logging.getLogger("mobicrowd.auth")


# =============================================================================
# Authentication response helpers
# =============================================================================

def _error_response(
    code: str,
    message: str,
    status_code: int,
):
    """
    Return one predictable authentication error contract.

    ``msg`` is retained for compatibility with existing frontend callers.
    New frontend code should use ``code`` for branching and ``message`` for
    display.
    """
    return Response(
        {
            "code": code,
            "message": message,
            "msg": message,
        },
        status=status_code,
    )


def _is_admin(user) -> bool:
    return bool(
        getattr(user, "is_superuser", False)
        or getattr(user, "role", "") == "Admin"
    )


def _kind_from_platform(platform: str) -> str:
    platform = (platform or "").strip().lower()

    return (
        "DASHBOARD"
        if platform == "web"
        else "APP"
    )


def _issue_tokens(
    user,
    *,
    sid: str,
    kind: str,
    device_uid: str,
    remember_me: bool = False,
):
    """
    Create the normal SimpleJWT refresh/access pair and bind both tokens
    to the Takleef server-side device session.

    Important:
    This function only creates the tokens.

    auth_success_response() decides whether:
      - they are returned in JSON (development/native), or
      - stored in HttpOnly cookies (production web).
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
# FCM helpers used by existing session/takeover logic
# =============================================================================

def _mask(
    s: str,
    keep_start: int = 10,
    keep_end: int = 6,
) -> str:
    s = (s or "").strip()

    if not s:
        return ""

    if len(s) <= keep_start + keep_end + 3:
        return s[:3] + "…"

    return (
        f"{s[:keep_start]}"
        f"…"
        f"{s[-keep_end:]}"
    )


def _response_has_fcm_error(
    resp,
    code: str,
) -> bool:
    try:
        j = resp.json()

        details = (
            (j.get("error", {}) or {})
            .get("details", [])
            or []
        )

        for detail in details:
            if (
                isinstance(detail, dict)
                and detail.get("errorCode") == code
            ):
                return True

    except Exception:
        pass

    return False


def _get_google_oauth_token() -> str:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request

    scopes = [
        "https://www.googleapis.com/auth/firebase.messaging"
    ]

    b64 = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT_B64"
    )

    if b64:
        info = json.loads(
            base64.b64decode(b64).decode("utf-8")
        )

        creds = (
            service_account
            .Credentials
            .from_service_account_info(
                info,
                scopes=scopes,
            )
        )

    else:
        path = getattr(
            settings,
            "GOOGLE_APPLICATION_CREDENTIALS",
            None,
        )

        if not path:
            raise RuntimeError(
                "Set FIREBASE_SERVICE_ACCOUNT_B64 "
                "or GOOGLE_APPLICATION_CREDENTIALS"
            )

        creds = (
            service_account
            .Credentials
            .from_service_account_file(
                path,
                scopes=scopes,
            )
        )

    creds.refresh(Request())

    return creds.token


def _get_project_id_from_sa():
    try:
        b64 = os.getenv(
            "FIREBASE_SERVICE_ACCOUNT_B64"
        )

        if b64:
            return json.loads(
                base64.b64decode(b64)
                .decode("utf-8")
            ).get("project_id")

        path = getattr(
            settings,
            "GOOGLE_APPLICATION_CREDENTIALS",
            None,
        )

        if path:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f).get(
                    "project_id"
                )

    except Exception:
        pass

    return None


def _probe_fcm_token_unregistered(
    token: str,
) -> bool:
    token = (token or "").strip()

    if not token:
        return False

    project_id = (
        getattr(
            settings,
            "FCM_PROJECT_ID",
            None,
        )
        or _get_project_id_from_sa()
    )

    if not project_id:
        logger.warning(
            "FCM probe skipped: missing "
            "FCM_PROJECT_ID / service account project_id"
        )
        return False

    try:
        access_token = (
            _get_google_oauth_token()
        )

    except Exception as exc:
        logger.warning(
            "FCM probe skipped: oauth token error: %s",
            exc,
        )
        return False

    url = (
        "https://fcm.googleapis.com/v1/projects/"
        f"{project_id}/messages:send"
    )

    headers = {
        "Authorization": (
            f"Bearer {access_token}"
        ),
        "Content-Type": "application/json",
    }

    body = {
        "validate_only": True,
        "message": {
            "token": token,
            "data": {
                "type": "SESSION_PROBE",
            },
        },
    }

    try:
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(body),
            timeout=10,
        )

    except Exception as exc:
        logger.warning(
            "FCM probe request failed: %s",
            exc,
        )
        return False

    if resp.ok:
        return False

    if (
        resp.status_code in (404, 410)
        or _response_has_fcm_error(
            resp,
            "UNREGISTERED",
        )
    ):
        return True

    logger.warning(
        "FCM probe error, not unlocking. "
        "code=%s body=%s",
        resp.status_code,
        resp.text.strip(),
    )

    return False


# =============================================================================
# Login
# =============================================================================

class LoginView(APIView):

    authentication_classes = []
    permission_classes = []

    def post(self, request):
        # ---------------------------------------------------------------------
        # Request
        # ---------------------------------------------------------------------

        email = str(
            request.data.get("email") or ""
        ).strip()

        password = request.data.get(
            "password"
        )

        device_uid = (
            request.data.get("device_uid")
            or ""
        ).strip()

        platform = (
            request.data.get("platform")
            or ""
        ).strip().lower()

        device_label = (
            request.data.get("device_label")
            or ""
        ).strip()

        takeover_method = (
            request.data.get("takeover_method")
            or ""
        ).strip()

        remember_me = coerce_bool(
            request.data.get("remember_me")
        )

        # ---------------------------------------------------------------------
        # Basic validation
        # ---------------------------------------------------------------------

        if not email or not password:
            return _error_response(
                "CREDENTIALS_REQUIRED",
                "Email and password are required.",
                status.HTTP_400_BAD_REQUEST,
            )

        if not device_uid:
            return _error_response(
                "DEVICE_UID_REQUIRED",
                "A device identifier is required.",
                status.HTTP_400_BAD_REQUEST,
            )

        if platform not in (
            "android",
            "ios",
            "web",
        ):
            return _error_response(
                "INVALID_PLATFORM",
                "Platform must be android, ios, or web.",
                status.HTTP_400_BAD_REQUEST,
            )

        # ---------------------------------------------------------------------
        # Validate account
        # ---------------------------------------------------------------------

        # Django's default authentication backend rejects inactive users and
        # returns None. Resolve the account first so that we can distinguish
        # invalid credentials from verification/approval states.
        account = (
            User.objects
            .filter(
                email__iexact=email
            )
            .first()
        )

        # Do not reveal whether the account exists.
        if (
            account is None
            or not account.check_password(
                password
            )
        ):
            return _error_response(
                "INVALID_CREDENTIALS",
                "The email or password is incorrect.",
                status.HTTP_401_UNAUTHORIZED,
            )

        # ---------------------------------------------------------------------
        # Inactive account states
        # ---------------------------------------------------------------------

        if not account.is_active:
            requester = (
                Requester.objects
                .filter(user=account)
                .only("approved")
                .first()
            )

            if (
                requester is not None
                and not requester.approved
            ):
                return _error_response(
                    "ACCOUNT_PENDING_APPROVAL",
                    (
                        "Your account is pending "
                        "administrator approval."
                    ),
                    status.HTTP_403_FORBIDDEN,
                )

            if (
                requester is not None
                and requester.approved
            ):
                return _error_response(
                    "EMAIL_VERIFICATION_REQUIRED",
                    (
                        "Your account is approved. "
                        "Check your email to activate it."
                    ),
                    status.HTTP_403_FORBIDDEN,
                )

            if getattr(
                account,
                "is_worker",
                False,
            ):
                return _error_response(
                    "OTP_VERIFICATION_REQUIRED",
                    (
                        "Please verify your account "
                        "using the OTP sent to your email."
                    ),
                    status.HTTP_403_FORBIDDEN,
                )

            return _error_response(
                "ACCOUNT_INACTIVE",
                (
                    "This account is inactive. "
                    "Please contact support."
                ),
                status.HTTP_403_FORBIDDEN,
            )

        # ---------------------------------------------------------------------
        # Django authentication
        # ---------------------------------------------------------------------

        user = authenticate(
            request,
            email=account.email,
            password=password,
        )

        if user is None:
            return _error_response(
                "INVALID_CREDENTIALS",
                "The email or password is incorrect.",
                status.HTTP_401_UNAUTHORIZED,
            )

        kind = _kind_from_platform(
            platform
        )

        now = timezone.now()

        # ---------------------------------------------------------------------
        # Single-session / takeover enforcement
        # ---------------------------------------------------------------------

        if not _is_admin(user):

            # -------------------------------------------------------------
            # 1. Release stale sessions
            # -------------------------------------------------------------

            released_count = (
                deactivate_stale_sessions(
                    user,
                    kind,
                )
            )

            if released_count:
                logger.info(
                    "LOGIN_STALE_RELEASED "
                    "user_id=%s kind=%s count=%s",
                    user.id,
                    kind,
                    released_count,
                )

            qs_active = (
                DeviceToken.objects
                .filter(
                    user=user,
                    kind=kind,
                    session_active=True,
                )
            )

            # -------------------------------------------------------------
            # 2. Same device: normal rotation is allowed
            # -------------------------------------------------------------

            same_device = (
                qs_active
                .filter(
                    device_uid=device_uid
                )
                .first()
            )

            if same_device:
                logger.info(
                    "LOGIN_SAME_DEVICE_ROTATION "
                    "user_id=%s kind=%s device_uid=%s",
                    user.id,
                    kind,
                    device_uid,
                )

            # -------------------------------------------------------------
            # 3. Check other active sessions
            # -------------------------------------------------------------

            others = list(
                qs_active
                .exclude(
                    device_uid=device_uid
                )
                .order_by(
                    "-session_last_seen",
                    "-last_seen",
                    "-created_at",
                )[:5]
            )

            if others:

                # ---------------------------------------------------------
                # 3.a Mobile session cleanup using FCM uninstall evidence
                # ---------------------------------------------------------

                fcm_stale_ids = []

                for row in others:
                    if (
                        row.platform
                        in ("android", "ios")
                        and _probe_fcm_token_unregistered(
                            getattr(
                                row,
                                "token",
                                "",
                            )
                        )
                    ):
                        fcm_stale_ids.append(
                            row.id
                        )

                if fcm_stale_ids:
                    (
                        DeviceToken.objects
                        .filter(
                            id__in=fcm_stale_ids
                        )
                        .update(
                            session_active=False,
                            session_sid="",
                            is_active=False,
                        )
                    )

                    logger.info(
                        "LOGIN_FCM_UNREGISTERED_RELEASED "
                        "user_id=%s kind=%s rows=%s",
                        user.id,
                        kind,
                        fcm_stale_ids,
                    )

                # Re-query after cleanup.
                qs_active = (
                    DeviceToken.objects
                    .filter(
                        user=user,
                        kind=kind,
                        session_active=True,
                    )
                )

                others = list(
                    qs_active
                    .exclude(
                        device_uid=device_uid
                    )
                    .order_by(
                        "-session_last_seen",
                        "-last_seen",
                        "-created_at",
                    )[:5]
                )

                # ---------------------------------------------------------
                # 3.b Another live session still exists
                # ---------------------------------------------------------

                if others:

                    # -----------------------------------------------------
                    # Email OTP takeover
                    # -----------------------------------------------------

                    if (
                        takeover_method
                        == "EMAIL_OTP"
                    ):
                        (
                            challenge,
                            raw_otp,
                        ) = create_email_otp_challenge(
                            user=user,
                            kind=kind,
                            platform=platform,
                            incoming_device_uid=(
                                device_uid
                            ),
                            incoming_device_label=(
                                device_label
                            ),
                        )

                        send_login_takeover_otp_email(
                            user,
                            raw_otp,
                            device_label=device_label,
                            platform=platform,
                            expires_minutes=10,
                        )

                        return Response(
                            {
                                "code": (
                                    "TAKEOVER_OTP_SENT"
                                ),
                                "message": (
                                    "This account is active "
                                    "elsewhere. We sent a "
                                    "verification code to "
                                    "your email."
                                ),
                                "msg": (
                                    "This account is active "
                                    "elsewhere. We sent a "
                                    "verification code to "
                                    "your email."
                                ),
                                "challenge_id": (
                                    challenge.challenge_id
                                ),
                                "expires_at": (
                                    challenge.expires_at
                                ),
                            },
                            status=(
                                status
                                .HTTP_409_CONFLICT
                            ),
                        )

                    # -----------------------------------------------------
                    # Existing-session approval takeover
                    # -----------------------------------------------------

                    if (
                        takeover_method
                        == "ACTIVE_SESSION"
                    ):
                        challenge = (
                            create_active_session_challenge(
                                user=user,
                                kind=kind,
                                platform=platform,
                                incoming_device_uid=(
                                    device_uid
                                ),
                                incoming_device_label=(
                                    device_label
                                ),
                            )
                        )

                        return Response(
                            {
                                "code": (
                                    "TAKEOVER_APPROVAL_REQUIRED"
                                ),
                                "message": (
                                    "Approve this login from "
                                    "your active session."
                                ),
                                "msg": (
                                    "Approve this login from "
                                    "your active session."
                                ),
                                "challenge_id": (
                                    challenge.challenge_id
                                ),
                                "expires_at": (
                                    challenge.expires_at
                                ),
                            },
                            status=(
                                status
                                .HTTP_409_CONFLICT
                            ),
                        )

                    # -----------------------------------------------------
                    # Tell frontend which takeover methods are available
                    # -----------------------------------------------------

                    row = others[0]

                    return Response(
                        {
                            "code": (
                                "TAKEOVER_VERIFICATION_REQUIRED"
                            ),
                            "message": (
                                "This account is active on "
                                "another device. Verify your "
                                "identity to continue."
                            ),
                            "msg": (
                                "This account is active on "
                                "another device. Verify your "
                                "identity to continue."
                            ),
                            "methods": [
                                "ACTIVE_SESSION",
                                "EMAIL_OTP",
                            ],
                            "kind": kind,
                            "last_seen": (
                                row.session_last_seen
                            ),
                        },
                        status=(
                            status.HTTP_409_CONFLICT
                        ),
                    )

        # ---------------------------------------------------------------------
        # 4. Create/rotate Takleef session
        # ---------------------------------------------------------------------

        sid = secrets.token_urlsafe(24)

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

        # ---------------------------------------------------------------------
        # 5. Issue JWT pair
        # ---------------------------------------------------------------------

        tokens = _issue_tokens(
            user,
            sid=sid,
            kind=kind,
            device_uid=device_uid,
            remember_me=remember_me,
        )

        # ---------------------------------------------------------------------
        # 6. Deliver authentication
        #
        # Development/native:
        #   access + refresh remain in JSON.
        #
        # Production web:
        #   JWTs are moved into HttpOnly Secure cookies and are NOT exposed
        #   in the JSON response.
        # ---------------------------------------------------------------------

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