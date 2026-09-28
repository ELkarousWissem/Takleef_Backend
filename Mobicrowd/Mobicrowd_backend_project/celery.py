# # Mobicrowd_backend_project/celery.py
# from __future__ import absolute_import, unicode_literals
#
# import os
# import multiprocessing
# from celery import Celery
# from celery.schedules import crontab
# from celery.utils.log import get_task_logger
#
# # Keep spawn if you rely on it (macOS/Windows etc.)
# multiprocessing.set_start_method("spawn", force=True)
#
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Mobicrowd_backend_project.settings")
#
# app = Celery("Mobicrowd_backend_project")
#
# # Pull CELERY_* settings from Django settings.py
# app.config_from_object("django.conf:settings", namespace="CELERY")
#
# # Make sure the tasks module is imported (your code is in mobicrowd/image_tasks.py)
# app.conf.imports = (
#     "mobicrowd.tasks",
# )
#
# # Single source of truth for Beat schedule
# app.conf.beat_schedule = {
#     # Backfill for time-based event reminders (80% elapsed + T-30m)
#     "reminders-backfill-every-5-min": {
#         "task": "mobicrowd.tasks.backfill_event_reminders",
#         "schedule": crontab(minute="*/5"),
#     },
#     # Backfill for worker inactivity (halfway, no submissions yet)
#     "backfill-worker-inactivity-every-5m": {
#         "task": "mobicrowd.tasks.backfill_worker_inactivity_checks",
#         "schedule": crontab(minute="*/5"),
#     },
# }
#
# # Discover appname.tasks as well (harmless to keep with explicit imports)
# app.autodiscover_tasks()
#
# logger = get_task_logger(__name__)
#
# @app.task(bind=True)
# def debug_task(self):
#     logger.info("Request: %r", self.request)
# Mobicrowd_backend_project/celery.py
from __future__ import absolute_import, unicode_literals

import os
import multiprocessing
from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger

# Keep spawn if you rely on it (e.g., TensorFlow / certain platforms).
multiprocessing.set_start_method("spawn", force=True)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "Mobicrowd_backend_project.settings.dev",
)

app = Celery("Mobicrowd_backend_project")

# Pull CELERY_* settings from Django settings.py
app.config_from_object("django.conf:settings", namespace="CELERY")

# Explicit imports (your tasks live here)
app.conf.imports = (
    "mobicrowd.tasks",
)

# Beat schedule (used when you run worker with -B, or when you run celery beat separately)
# We keep the cadence at 1 minute so reminders/end/inactivity are caught quickly.
app.conf.beat_schedule = {
    # Backfill for time-based event reminders (80% elapsed + T-30m)
    "reminders-backfill-every-1-min": {
        "task": "mobicrowd.tasks.backfill_event_reminders",
        "schedule": crontab(minute="*/1"),
    },
    # Backfill for end-of-event notifications
    "end-backfill-every-1-min": {
        "task": "mobicrowd.tasks.backfill_event_end_notifications",
        "schedule": crontab(minute="*/1"),
    },
    # Backfill for worker inactivity (halfway, no submissions yet)
    "worker-inactivity-backfill-every-1-min": {
        "task": "mobicrowd.tasks.backfill_worker_inactivity_checks",
        "schedule": crontab(minute="*/1"),
    },
}

# Discover appname.tasks as well (harmless to keep with explicit imports)
app.autodiscover_tasks()

logger = get_task_logger(__name__)

@app.task(bind=True)
def debug_task(self):
    logger.info("Request: %r", self.request)