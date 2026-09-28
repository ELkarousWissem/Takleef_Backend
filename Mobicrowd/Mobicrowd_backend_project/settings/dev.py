"""
Development settings.

Whenever this module is selected, it loads the local development
environment from:

    <project-root>/env/.env.dev

Existing process environment variables always take precedence.
"""

import os
from pathlib import Path


# ============================================================================
# Local development environment loader
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_ENV_FILE = PROJECT_ROOT / "env" / ".env.dev"


def _load_local_env(path: Path) -> None:
    """
    Load KEY=VALUE pairs from env/.env.dev.

    Existing environment variables are never overwritten.
    This loader is development-only.
    """

    if not path.is_file():
        raise RuntimeError(
            f"Local development environment file not found: {path}"
        )

    for raw_line in path.read_text(encoding="utf-8").splitlines():

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].lstrip()

        key, separator, value = line.partition("=")

        if not separator:
            continue

        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Strip one matching pair of outer quotes.
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        # Explicit shell/container environment wins over .env.dev.
        os.environ.setdefault(key, value)


_load_local_env(LOCAL_ENV_FILE)


# IMPORTANT:
# base.py must only be imported AFTER env/.env.dev has been loaded,
# because base.py requires DJANGO_SECRET_KEY and DB_PASSWORD.
from .base import *  # noqa: F401,F403,E402


# ============================================================================
# Development Django behavior
# ============================================================================

DEBUG = True


ALLOWED_HOSTS = env_list(
    "ALLOWED_HOSTS",
    [
        "localhost",
        "127.0.0.1",
    ],
)


# Support both current and older django-cors-headers setting names.
CORS_ALLOW_ALL_ORIGINS = True
CORS_ORIGIN_ALLOW_ALL = True


CSRF_TRUSTED_ORIGINS = env_list(
    "CSRF_TRUSTED_ORIGINS",
    [
        "http://192.168.100.44:4200",
        "http://127.0.0.1:4200",
    ],
)


BASE_URL = os.environ.get(
    "BASE_URL",
    "http://192.168.100.44:8002",
)

FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "http://192.168.100.44:4200",
)


# ============================================================================
# Development security behavior
# ============================================================================

SECURE_SSL_REDIRECT = False

SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False


# ============================================================================
# Development logging
# ============================================================================

LOGGING["handlers"]["console"]["level"] = "DEBUG"

LOGGING["loggers"]["django"]["level"] = os.environ.get(
    "DJANGO_LOG_LEVEL",
    "INFO",
).upper()

LOGGING["loggers"]["celery"]["level"] = os.environ.get(
    "CELERY_LOG_LEVEL",
    "DEBUG",
).upper()

LOGGING["loggers"]["mobicrowd"]["level"] = os.environ.get(
    "APP_LOG_LEVEL",
    "DEBUG",
).upper()

JWT_COOKIE_AUTH_ENABLED = False
JWT_COOKIE_SECURE = False