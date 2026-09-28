# mobicrowd/ws_auth.py

from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.db import close_old_connections

from mobicrowd.authentication.cookie_auth import (
    access_cookie_name,
)
from mobicrowd.authentication.session_auth import (
    SessionBoundJWTAuthentication,
)


class QueryStringJWTAuthMiddleware(BaseMiddleware):

    def _cookie_token(self, scope):
        if not getattr(
            settings,
            "JWT_COOKIE_AUTH_ENABLED",
            False,
        ):
            return None

        headers = dict(
            scope.get("headers", [])
        )

        raw_cookie = headers.get(
            b"cookie",
            b"",
        ).decode("latin1")

        if not raw_cookie:
            return None

        cookie = SimpleCookie()

        try:
            cookie.load(raw_cookie)
        except Exception:
            return None

        item = cookie.get(access_cookie_name())

        return item.value if item else None

    async def __call__(
        self,
        scope,
        receive,
        send,
    ):
        close_old_connections()

        # Production browser path:
        # authenticate from HttpOnly cookie.
        token = self._cookie_token(scope)

        # Backward-compatible development/native path.
        if not token:
            query = parse_qs(
                scope.get(
                    "query_string",
                    b"",
                ).decode()
            )

            vals = query.get("token")

            if vals:
                token = vals[0]

        # Non-browser WebSocket clients may still send Bearer.
        if not token:
            headers = dict(
                scope.get("headers", [])
            )

            auth = headers.get(
                b"authorization",
                b"",
            ).decode()

            if auth.lower().startswith("bearer "):
                token = auth.split(
                    " ",
                    1,
                )[1]

        user = AnonymousUser()

        if token:
            auth = SessionBoundJWTAuthentication()

            try:
                validated = auth.get_validated_token(
                    token
                )

                user = await database_sync_to_async(
                    auth.get_user
                )(validated)

            except Exception:
                user = AnonymousUser()

        scope["user"] = user

        return await super().__call__(
            scope,
            receive,
            send,
        )