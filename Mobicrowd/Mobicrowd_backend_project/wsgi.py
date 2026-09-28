"""
WSGI config for Mobicrowd_backend_project.
"""

import os

from django.core.wsgi import get_wsgi_application


# Local/manual WSGI execution defaults to development.
# Production already exports settings.prod before importing WSGI.
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "Mobicrowd_backend_project.settings.dev",
)

application = get_wsgi_application()