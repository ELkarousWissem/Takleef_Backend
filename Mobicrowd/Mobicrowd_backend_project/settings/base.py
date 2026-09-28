
"""
Shared Django settings for Mobicrowd_backend_project.

This file contains only settings shared by development and production.
Environment-specific behavior belongs in dev.py and prod.py.

Secrets are never hard-coded. They are read from environment variables.
"""

from datetime import timedelta
from pathlib import Path
import os

from django.core.exceptions import ImproperlyConfigured


# =============================================================================
# Environment helpers
# =============================================================================

def env_required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ImproperlyConfigured(
            f"Required environment variable '{name}' is not set."
        )
    return value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ImproperlyConfigured(
            f"Environment variable '{name}' must be an integer."
        ) from exc


def env_list(name: str, default=None):
    """
    Parse comma-separated environment values.

    Example:
      ALLOWED_HOSTS=takleef.ai,www.takleef.ai
    """
    value = os.environ.get(name)
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


# =============================================================================
# Paths
# =============================================================================

# base.py lives at:
#   <project-root>/Mobicrowd_backend_project/settings/base.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

EMAIL_LOGO_PATH = BASE_DIR / "templates" / "takleef.png"


# =============================================================================
# Core Django security
# =============================================================================

SECRET_KEY = env_required("DJANGO_SECRET_KEY")

# DEBUG and ALLOWED_HOSTS intentionally live in dev.py / prod.py.


# =============================================================================
# Applications
# =============================================================================

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "mobicrowd",
    "corsheaders",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "storages",
    "channels",
]


# =============================================================================
# Middleware
# =============================================================================

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    # django-cors-headers recommends CorsMiddleware before CommonMiddleware.
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]


# =============================================================================
# URL / WSGI / ASGI
# =============================================================================

ROOT_URLCONF = "Mobicrowd_backend_project.urls"
WSGI_APPLICATION = "Mobicrowd_backend_project.wsgi.application"
ASGI_APPLICATION = "Mobicrowd_backend_project.asgi.application"


# =============================================================================
# Templates
# =============================================================================

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# =============================================================================
# Database
# =============================================================================

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.environ.get("DB_NAME", "mobicrowd"),
        "USER": os.environ.get("DB_USER", "mobicrowd_user"),
        "PASSWORD": env_required("DB_PASSWORD"),
        "HOST": os.environ.get("DB_HOST", "127.0.0.1"),
        "PORT": os.environ.get("DB_PORT", "3306"),
        "CONN_MAX_AGE": env_int("DB_CONN_MAX_AGE", 60),
    }
}


# =============================================================================
# Redis / Channels / Cache
# =============================================================================

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = env_int("REDIS_PORT", 6379)

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [f"redis://{REDIS_HOST}:{REDIS_PORT}/1"],
        },
    }
}

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": f"redis://{REDIS_HOST}:{REDIS_PORT}/2",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
        "TIMEOUT": None,
    }
}


# =============================================================================
# Django REST Framework
# =============================================================================

REST_FRAMEWORK = {
    "DEFAULT_THROTTLE_RATES": {
        "nominatim": "1/second",
    },
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "mobicrowd.authentication.session_auth.SessionBoundJWTAuthentication",
    ),
}


# =============================================================================
# JWT
# =============================================================================

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=365),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": False,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "VERIFYING_KEY": None,
    "AUTH_HEADER_TYPES": ("Bearer",),
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
    "TOKEN_TYPE_CLAIM": "token_type",
}

DASHBOARD_SESSION_TIMEOUT_MINUTES = None
APP_SESSION_TIMEOUT_MINUTES = None


# =============================================================================
# Password validation
# =============================================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        ),
    },
]


# =============================================================================
# Auth / account
# =============================================================================

AUTH_USER_MODEL = "mobicrowd.User"

ACCOUNT_AUTHENTICATION_METHOD = "email"
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_EMAIL_VERIFICATION = "mandatory"
ACCOUNT_LOGIN_ATTEMPTS_LIMIT = 5
ACCOUNT_LOGIN_ATTEMPTS_TIMEOUT = 300

PASSWORD_RESET_TIMEOUT = 3600


# =============================================================================
# Internationalization
# =============================================================================

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

NOMINATIM_USER_AGENT = os.environ.get(
    "NOMINATIM_USER_AGENT",
    "Takleef/1.0",
)


# =============================================================================
# Static files
# =============================================================================

STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# =============================================================================
# Email
# =============================================================================

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = env_int("EMAIL_PORT", 587)
EMAIL_USE_TLS = env_bool("EMAIL_USE_TLS", True)
EMAIL_USE_SSL = env_bool("EMAIL_USE_SSL", False)
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")


# =============================================================================
# Firebase / Google
# =============================================================================

FCM_PROJECT_ID = os.environ.get("FCM_PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS"
)


# =============================================================================
# AWS / S3
# =============================================================================

# On OVH/VPS these values can be injected from the VPS environment.
# On AWS compute, prefer an IAM role and leave explicit key variables unset.
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")

AWS_STORAGE_BUCKET_NAME = os.environ.get(
    "AWS_STORAGE_BUCKET_NAME",
    "takleef-eu",
)
AWS_S3_REGION_NAME = os.environ.get(
    "AWS_S3_REGION_NAME",
    "eu-central-1",
)
AWS_LOCATION = os.environ.get("AWS_LOCATION", "multimedia")

AWS_S3_CUSTOM_DOMAIN = (
    f"{AWS_STORAGE_BUCKET_NAME}.s3."
    f"{AWS_S3_REGION_NAME}.amazonaws.com"
)

AWS_DEFAULT_ACL = None
AWS_QUERYSTRING_AUTH = True
AWS_S3_FILE_OVERWRITE = False

# Preserved to match the current project configuration.
DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
MEDIA_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/{AWS_LOCATION}/"


# =============================================================================
# Celery
# =============================================================================

CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
CELERY_RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

CELERY_TASK_IGNORE_RESULT = False
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"
CELERY_ENABLE_UTC = True
CELERY_TASK_SOFT_TIME_LIMIT = 300
CELERY_TASK_TIME_LIMIT = 360

CELERY_BROKER_TRANSPORT_OPTIONS = {
    "visibility_timeout": 600,
}


# =============================================================================
# OpenRouter credentials/configuration
# =============================================================================

# The key file must be outside the Git repository in production and mounted
# read-only into the container.
OPENROUTER_KEYS_FILE = os.environ.get("OPENROUTER_KEYS_FILE")

OPENROUTER_IMAGE_RELEVANCE_MODEL = "qwen/qwen3-vl-8b-instruct"
OPENROUTER_IMAGE_RELEVANCE_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_VIDEO_RELEVANCE_MODEL = "qwen/qwen3-vl-8b-instruct"
OPENROUTER_VIDEO_RELEVANCE_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_OCR_MODEL = "qwen/qwen3-vl-8b-instruct"
OPENROUTER_OCR_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_TASK_UNDERSTANDING_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_TASK_UNDERSTANDING_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_TARGET_EXPANSION_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_TARGET_EXPANSION_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_DESCRIPTION_ENHANCEMENT_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_DESCRIPTION_ENHANCEMENT_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_TRANSLATION_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
OPENROUTER_TRANSLATION_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_LANGUAGE_DETECTION_MODEL = "meta-llama/llama-3.2-3b-instruct"
OPENROUTER_LANGUAGE_DETECTION_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_TEXTUAL_PIPELINE_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_TEXTUAL_PIPELINE_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_ANNOTATION_MODEL = "qwen/qwen3-vl-8b-instruct"
OPENROUTER_ANNOTATION_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_STORY_TELLER_MODEL = "qwen/qwen3.5-flash-02-23"

# Preserved as a string because that is the type used in the current settings.
OPENROUTER_STORY_TELLER_FALLBACK_MODELS = (
    "mistralai/mistral-small-3.2-24b-instruct"
)

OPENROUTER_TEXTUAL_SAFETY_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_TEXTUAL_SAFETY_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
]

OPENROUTER_OCR_MAX_ATTEMPTS = env_int(
    "OPENROUTER_OCR_MAX_ATTEMPTS",
    6,
)
OPENROUTER_OCR_MAX_TOKENS = env_int(
    "OPENROUTER_OCR_MAX_TOKENS",
    512,
)


# =============================================================================
# Qdrant
# =============================================================================

QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PORT = env_int("QDRANT_PORT", 6333)
QDRANT_COLLECTION = os.environ.get(
    "QDRANT_COLLECTION",
    "photo_embeddings",
)
QDRANT_COLLECTION_VIDEO = os.environ.get(
    "QDRANT_COLLECTION_VIDEO",
    "video_embeddings",
)


# =============================================================================
# Logging
# =============================================================================

DJANGO_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO").upper()
CELERY_LOG_LEVEL = os.environ.get("CELERY_LOG_LEVEL", "INFO").upper()
APP_LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "concise": {
            "format": "{levelname} {asctime} {name} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "concise",
            "level": "INFO",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": DJANGO_LOG_LEVEL,
            "propagate": False,
        },
        "celery": {
            "handlers": ["console"],
            "level": CELERY_LOG_LEVEL,
            "propagate": False,
        },
        "mobicrowd": {
            "handlers": ["console"],
            "level": APP_LOG_LEVEL,
            "propagate": False,
        },
    },
}