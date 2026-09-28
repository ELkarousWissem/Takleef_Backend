# mobicrowd/notify_helpers.py
from __future__ import annotations
from typing import Iterable
from django.db.models import QuerySet
from mobicrowd.models.Users import User, Worker
from mobicrowd.models.submisson import EventWorker, Event
from mobicrowd.notify import notify_user

def make_payload(*, type: str, **ids) -> dict:
    """
    Consistent, flexible payload:
    make_payload(type="join_request", event_id=1, status="approved", event_worker_id=9)
    make_payload(type="event", event_id=1)
    """
    return {"type": type, **ids}

def notify_admins(*, event_type: str, title: str, body: str = "", payload: dict | None = None, priority: str = "high"):
    admin_ids = list(User.objects.filter(role="Admin", is_active=True).values_list("id", flat=True))
    for uid in admin_ids:
        notify_user(uid, event_type=event_type, title=title, body=body, payload=payload or {}, priority=priority)

def notify_workers(*, workers: Iterable[Worker] | QuerySet[Worker], event_type: str, title: str, body: str = "", payload: dict | None = None, priority: str = "high"):
    for w in workers:
        notify_user(w.user_id, event_type=event_type, title=title, body=body, payload=payload or {}, priority=priority)

def notify_event_subscribers(event: Event, *, event_type: str, title: str, body: str = "", payload: dict | None = None, priority: str = "high"):
    qs = (EventWorker.objects
          .filter(event=event, status=EventWorker.APPROVED)
          .select_related("worker__user"))
    for ew in qs:
        notify_user(ew.worker.user_id, event_type=event_type, title=title, body=body, payload=payload or {}, priority=priority)
