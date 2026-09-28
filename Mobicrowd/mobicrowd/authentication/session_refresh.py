from django.conf import settings
from django.db import transaction
from django.utils import timezone

from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from rest_framework.response import Response

from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.serializers import TokenRefreshSerializer
from rest_framework_simplejwt.settings import api_settings
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView

from mobicrowd.authentication.cookie_auth import (
    refresh_cookie_name,
    set_auth_cookies,
)
from mobicrowd.models.notifications import DeviceToken


class SessionBoundTokenRefreshSerializer(TokenRefreshSerializer):
    """
    Refresh JWTs only while their server-side DeviceToken session
    remains active.
    """

    session_user = None

    def validate(self, attrs):
        raw_refresh = attrs.get("refresh")

        if not raw_refresh:
            raise InvalidToken(
                "Refresh token is required."
            )

        try:
            refresh = self.token_class(
                raw_refresh
            )
        except TokenError as exc:
            raise InvalidToken(
                str(exc)
            ) from exc

        user_id = refresh.get(
            api_settings.USER_ID_CLAIM
        )

        sid = refresh.get("sid")
        kind = refresh.get("kind")
        device_uid = refresh.get(
            "device_uid"
        )

        if (
            not user_id
            or not sid
            or kind not in (
                "APP",
                "DASHBOARD",
            )
            or not device_uid
        ):
            raise InvalidToken(
                "Refresh token is not bound "
                "to a valid session."
            )

        with transaction.atomic():
            row = (
                DeviceToken.objects
                .select_for_update()
                .select_related("user")
                .filter(
                    user_id=user_id,
                    device_uid=device_uid,
                    kind=kind,
                    session_sid=sid,
                    session_active=True,
                )
                .first()
            )

            if row is None:
                raise InvalidToken(
                    "Session is no longer active."
                )

            if not getattr(
                row.user,
                "is_active",
                False,
            ):
                DeviceToken.objects.filter(
                    pk=row.pk
                ).update(
                    session_active=False,
                    session_sid="",
                    is_active=False,
                )

                raise InvalidToken(
                    "User account is inactive."
                )

            now = timezone.now()

            DeviceToken.objects.filter(
                pk=row.pk
            ).update(
                session_last_seen=now,
                last_seen=now,
                is_active=True,
            )

            self.session_user = row.user

            # Normal SimpleJWT refresh/rotation/blacklisting.
            return super().validate(attrs)


class SessionBoundTokenRefreshView(
    TokenRefreshView
):
    authentication_classes = []
    permission_classes = []
    serializer_class = (
        SessionBoundTokenRefreshSerializer
    )

    def post(
        self,
        request,
        *args,
        **kwargs,
    ):
        data = request.data.copy()

        # -------------------------------------------------------------
        # Development/native:
        # refresh token is still provided in JSON body.
        # -------------------------------------------------------------
        raw_refresh = str(
            data.get("refresh") or ""
        ).strip()

        cookie_mode = False

        # -------------------------------------------------------------
        # Production web:
        # refresh token comes from HttpOnly cookie.
        # -------------------------------------------------------------
        if (
            not raw_refresh
            and getattr(
                settings,
                "JWT_COOKIE_AUTH_ENABLED",
                False,
            )
        ):
            raw_refresh = str(
                request.COOKIES.get(
                    refresh_cookie_name()
                )
                or ""
            ).strip()

            if raw_refresh:
                cookie_mode = True

                # Refresh through a cookie is state-changing,
                # so enforce CSRF.
                SessionAuthentication().enforce_csrf(
                    request
                )

                data["refresh"] = raw_refresh

        if not raw_refresh:
            raise InvalidToken(
                "Refresh token is required."
            )

        serializer = self.get_serializer(
            data=data
        )

        serializer.is_valid(
            raise_exception=True
        )

        result = dict(
            serializer.validated_data
        )

        # -------------------------------------------------------------
        # Native/development behavior remains unchanged.
        # -------------------------------------------------------------
        if not cookie_mode:
            return Response(
                result,
                status=status.HTTP_200_OK,
            )

        # -------------------------------------------------------------
        # Production web.
        # Do NOT expose access/refresh JWTs in response JSON.
        # -------------------------------------------------------------
        access = result.get("access")

        if not access:
            raise InvalidToken(
                "Refresh did not issue "
                "an access token."
            )

        # ROTATE_REFRESH_TOKENS=True may produce a replacement
        # refresh token. Otherwise continue using the existing one.
        next_refresh = (
            result.get("refresh")
            or raw_refresh
        )

        remember_me = False

        try:
            refresh_obj = RefreshToken(
                next_refresh
            )

            remember_me = bool(
                refresh_obj.get(
                    "remember",
                    False,
                )
            )
        except Exception:
            pass

        user = getattr(
            serializer,
            "session_user",
            None,
        )

        response_payload = {
            "code": "SESSION_REFRESHED",
        }

        if user is not None:
            response_payload.update({
                "id": user.id,
                "role": getattr(
                    user,
                    "role",
                    "",
                ),
            })

        response = Response(
            response_payload,
            status=status.HTTP_200_OK,
        )

        return set_auth_cookies(
            request,
            response,
            access=access,
            refresh=next_refresh,
            remember_me=remember_me,
        )