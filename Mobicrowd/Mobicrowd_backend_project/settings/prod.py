"""
Production settings.

The production Docker entrypoint selects this module by default:

    DJANGO_ENV=prod
    -> Mobicrowd_backend_project.settings.prod
"""

import os

from django.core.exceptions import ImproperlyConfigured

from .base import *  # noqa: F401,F403


DEBUG = False


# =============================================================================
# Hosts / CORS / CSRF
# =============================================================================

ALLOWED_HOSTS = env_list(
    "ALLOWED_HOSTS",
    [
        "takleef.ai",
        ".takleef.ai",
        "localhost",
        "127.0.0.1",
    ],
)

# django-cors-headers
CORS_ALLOW_ALL_ORIGINS = False

# Compatibility with older django-cors-headers versions/configuration names.
CORS_ORIGIN_ALLOW_ALL = False

CORS_ALLOWED_ORIGINS = env_list(
    "CORS_ALLOWED_ORIGINS",
    [
        "https://app.takleef.ai",
        "https://dev.takleef.ai",
        "http://localhost",
        "https://localhost",
        "capacitor://localhost",
    ],
)

CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = env_list(
    "CSRF_TRUSTED_ORIGINS",
    [
        "https://app.takleef.ai",
        "https://dev.takleef.ai",
        "http://localhost",
        "https://localhost",
    ],
)


# =============================================================================
# Public URLs
# =============================================================================

BASE_URL = os.environ.get(
    "BASE_URL",
    "https://app.takleef.ai",
)

FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://app.takleef.ai",
)


# =============================================================================
# HTTPS / proxy security
# =============================================================================

# Nginx must send:
#   proxy_set_header X-Forwarded-Proto $scheme;
SECURE_PROXY_SSL_HEADER = (
    "HTTP_X_FORWARDED_PROTO",
    "https",
)

SECURE_SSL_REDIRECT = env_bool(
    "SECURE_SSL_REDIRECT",
    True,
)

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True

SESSION_COOKIE_SAMESITE = os.environ.get(
    "SESSION_COOKIE_SAMESITE",
    "None",
)

CSRF_COOKIE_SAMESITE = os.environ.get(
    "CSRF_COOKIE_SAMESITE",
    "None",
)

SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"

SECURE_HSTS_SECONDS = env_int(
    "SECURE_HSTS_SECONDS",
    0,
)

SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool(
    "SECURE_HSTS_INCLUDE_SUBDOMAINS",
    False,
)

SECURE_HSTS_PRELOAD = env_bool(
    "SECURE_HSTS_PRELOAD",
    False,
)


# =============================================================================
# Production logging
# =============================================================================

LOGGING["handlers"]["console"]["level"] = "WARNING"

LOGGING["loggers"]["django"]["level"] = os.environ.get(
    "DJANGO_LOG_LEVEL",
    "WARNING",
).upper()

LOGGING["loggers"]["celery"]["level"] = os.environ.get(
    "CELERY_LOG_LEVEL",
    "WARNING",
).upper()

LOGGING["loggers"]["mobicrowd"]["level"] = os.environ.get(
    "APP_LOG_LEVEL",
    "WARNING",
).upper()


# =============================================================================
# Production startup validation
# =============================================================================

required_environment = (
    "DJANGO_SECRET_KEY",
    "DB_PASSWORD",
    "EMAIL_HOST_USER",
    "EMAIL_HOST_PASSWORD",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "FCM_PROJECT_ID",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENROUTER_KEYS_FILE",
)

missing = [
    name
    for name in required_environment
    if not os.environ.get(name)
]

if missing:
    raise ImproperlyConfigured(
        "Missing required production environment variable(s): "
        + ", ".join(missing)
    )


# =============================================================================
# Production Web / APK JWT cookies
# =============================================================================

JWT_COOKIE_AUTH_ENABLED = True

JWT_ACCESS_COOKIE_NAME = "takleef_access"
JWT_REFRESH_COOKIE_NAME = "takleef_refresh"

JWT_COOKIE_SECURE = True

JWT_COOKIE_SAMESITE = os.environ.get(
    "JWT_COOKIE_SAMESITE",
    "None",
)

JWT_COOKIE_DOMAIN = (
    os.environ.get("JWT_COOKIE_DOMAIN") or None
)


# =============================================================================
# Angular / Django CSRF cookie convention
# =============================================================================

CSRF_COOKIE_NAME = "XSRF-TOKEN"
CSRF_HEADER_NAME = "HTTP_X_XSRF_TOKEN"

CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SECURE = True