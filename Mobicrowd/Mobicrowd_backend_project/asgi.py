# Mobicrowd_backend_project/asgi.py

import os


# Local/manual Daphne execution defaults to development.
#
# In production entrypoint.sh has already exported:
# Mobicrowd_backend_project.settings.prod
#
# setdefault() will therefore NOT overwrite production.
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "Mobicrowd_backend_project.settings.dev",
)


# Initialize Django first.
from django.core.asgi import get_asgi_application

django_asgi_app = get_asgi_application()


# Only after Django initialization import code that may touch models.
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.urls import path

from mobicrowd.ws_auth import QueryStringJWTAuthMiddleware
from mobicrowd.wsconsumers import NotificationConsumer


application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,

        "websocket": AllowedHostsOriginValidator(
            QueryStringJWTAuthMiddleware(
                URLRouter(
                    [
                        path(
                            "ws/notifications/",
                            NotificationConsumer.as_asgi(),
                        ),
                    ]
                )
            )
        ),
    }
)