from django.conf import settings
from django.middleware.csrf import get_token
from rest_framework.response import Response


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def cookie_auth_enabled() -> bool:
    return bool(
        getattr(settings, "JWT_COOKIE_AUTH_ENABLED", False)
    )


def web_cookie_auth_enabled(platform: str) -> bool:
    return (
        cookie_auth_enabled()
        and str(platform or "").strip().lower() == "web"
    )


def access_cookie_name() -> str:
    return getattr(
        settings,
        "JWT_ACCESS_COOKIE_NAME",
        "takleef_access",
    )


def refresh_cookie_name() -> str:
    return getattr(
        settings,
        "JWT_REFRESH_COOKIE_NAME",
        "takleef_refresh",
    )


def _cookie_kwargs():
    kwargs = {
        "httponly": True,
        "secure": bool(
            getattr(settings, "JWT_COOKIE_SECURE", False)
        ),
        "samesite": getattr(
            settings,
            "JWT_COOKIE_SAMESITE",
            "Lax",
        ),
        "path": "/",
    }

    domain = getattr(
        settings,
        "JWT_COOKIE_DOMAIN",
        None,
    )

    if domain:
        kwargs["domain"] = domain

    return kwargs


def set_auth_cookies(
    request,
    response,
    *,
    access: str,
    refresh: str,
    remember_me: bool,
):
    # Causes Django's CSRF middleware to issue the readable
    # XSRF token cookie.
    get_token(request)

    kwargs = _cookie_kwargs()

    if remember_me:
        access_seconds = int(
            settings.SIMPLE_JWT[
                "ACCESS_TOKEN_LIFETIME"
            ].total_seconds()
        )

        refresh_seconds = int(
            settings.SIMPLE_JWT[
                "REFRESH_TOKEN_LIFETIME"
            ].total_seconds()
        )

        response.set_cookie(
            access_cookie_name(),
            access,
            max_age=access_seconds,
            **kwargs,
        )

        response.set_cookie(
            refresh_cookie_name(),
            refresh,
            max_age=refresh_seconds,
            **kwargs,
        )

    else:
        # Session cookies: disappear when the browser session ends.
        response.set_cookie(
            access_cookie_name(),
            access,
            **kwargs,
        )

        response.set_cookie(
            refresh_cookie_name(),
            refresh,
            **kwargs,
        )

    return response


def clear_auth_cookies(response):
    domain = getattr(
        settings,
        "JWT_COOKIE_DOMAIN",
        None,
    )

    samesite = getattr(
        settings,
        "JWT_COOKIE_SAMESITE",
        "Lax",
    )

    response.delete_cookie(
        access_cookie_name(),
        path="/",
        domain=domain,
        samesite=samesite,
    )

    response.delete_cookie(
        refresh_cookie_name(),
        path="/",
        domain=domain,
        samesite=samesite,
    )

    return response


def auth_success_response(
    request,
    *,
    payload: dict,
    tokens: dict,
    platform: str,
    remember_me: bool,
    status_code: int,
):
    if not web_cookie_auth_enabled(platform):
        return Response(
            {
                **payload,
                **tokens,
            },
            status=status_code,
        )

    # Production web receives only the minimum UI metadata.
    response = Response(
        {
            **payload,
            "id": tokens["id"],
            "role": tokens["role"],
        },
        status=status_code,
    )

    return set_auth_cookies(
        request,
        response,
        access=tokens["access"],
        refresh=tokens["refresh"],
        remember_me=remember_me,
    )