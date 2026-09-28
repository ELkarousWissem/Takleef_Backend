from django.conf import settings
from django.utils import timezone
from rest_framework.authentication import SessionAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.authentication import JWTAuthentication

from mobicrowd.authentication.cookie_auth import access_cookie_name
from mobicrowd.authentication.session_security import is_session_stale
from mobicrowd.models.notifications import DeviceToken


class SessionBoundJWTAuthentication(JWTAuthentication):

    def authenticate(self, request):
        # Existing Bearer authentication remains authoritative for
        # development and native Android/iOS.
        header = self.get_header(request)

        if header is not None:
            return super().authenticate(request)

        if not getattr(
            settings,
            "JWT_COOKIE_AUTH_ENABLED",
            False,
        ):
            return None

        raw_token = request.COOKIES.get(
            access_cookie_name()
        )

        if not raw_token:
            return None

        validated_token = self.get_validated_token(
            raw_token
        )

        user = self.get_user(validated_token)

        # Cookie authentication requires CSRF protection for
        # unsafe HTTP methods.
        SessionAuthentication().enforce_csrf(request)

        return user, validated_token

    def get_user(self, validated_token):
        user = super().get_user(validated_token)

        sid = validated_token.get("sid")
        kind = validated_token.get("kind")
        device_uid = validated_token.get("device_uid")

        if (
            not sid
            or kind not in ("APP", "DASHBOARD")
            or not device_uid
        ):
            raise AuthenticationFailed(
                "Session not bound."
            )

        row = DeviceToken.objects.filter(
            user=user,
            device_uid=device_uid,
            kind=kind,
            session_sid=sid,
            session_active=True,
        ).first()

        if not row:
            raise AuthenticationFailed(
                "Session is no longer active."
            )

        now = timezone.now()

        if is_session_stale(row, now):
            DeviceToken.objects.filter(
                pk=row.pk
            ).update(
                session_active=False,
                session_sid="",
                is_active=False,
            )

            raise AuthenticationFailed(
                "Session expired."
            )

        DeviceToken.objects.filter(
            pk=row.pk
        ).update(
            session_last_seen=now,
            last_seen=now,
        )

        return user