# mobicrowd/notify.py
from __future__ import annotations
import os, json, base64, logging
import uuid
from datetime import timedelta
from typing import Optional, TypedDict, Sequence
import requests
from django.conf import settings
from mobicrowd.models.notifications import Notification, DeviceToken

logger = logging.getLogger("mobicrowd")

# ---------- Public API ----------
def notify_user(
    user_id: int,
    *,
    event_type: str,
    title: str,
    body: str = "",
    payload: dict | None = None,
    priority: str = "normal",
) -> None:
    payload = payload or {}

    # 1) Persist
    # --- store emoji-safe aliases in DB ---
    n = Notification.objects.create(
        user_id=user_id,
        event_type=event_type,
        title=encode_for_db(title),  # <-- encode
        body=encode_for_db(body),  # <-- encode
        payload=payload,
        priority=priority,
    )

    # 2) Realtime
    # --- realtime WS: send real emoji to clients ---
    try:
        async_to_sync(get_channel_layer().group_send)(
            f"user_{user_id}",
            {
                "type": "notify",
                "message": {
                    "kind": "new_notification",
                    "notification": {
                        "id": n.id,
                        "event_type": event_type,
                        "title": decode_for_api(n.title),  # <-- decode
                        "body": decode_for_api(n.body),  # <-- decode
                        "payload": payload,
                        "priority": priority,
                        "created_at": n.created_at.isoformat(),
                    },
                },
            },
        )
    except Exception as e:
        logger.warning("WS fanout failed for user %s: %s", user_id, e)

        # --- push (your working FCM v1 code or legacy) ---
    try:
        _push_fcm_v1(user_id, decode_for_api(n.title), decode_for_api(n.body),
                  data={"notification_id": n.id, **payload})
    except Exception as e:
        logger.error("FCM push failed for user %s: %s", user_id, e)

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


logger = logging.getLogger("ws")

def signal_user(*, user_id: int, signal: str, payload: dict | None = None) -> str:
    """
    WS-only silent signal.
    Returns correlation id to trace the signal end-to-end.
    """
    payload = payload or {}
    corr_id = uuid.uuid4().hex[:10]   # short correlation id

    message = {
        "kind": "silent_signal",
        "signal": signal,
        "corr_id": corr_id,
        **payload,
    }

    group = f"user_{user_id}"
    logger.info("[WS] enqueue start corr=%s group=%s signal=%s payload_keys=%s",
                corr_id, group, signal, list(payload.keys()))

    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            group,
            {
                "type": "notify",    # MUST match consumer method name
                "message": message,  # consumer sends this JSON
            },
        )
        logger.info("[WS] enqueue OK corr=%s group=%s", corr_id, group)
    except Exception:
        logger.exception("[WS] enqueue FAILED corr=%s group=%s", corr_id, group)
        raise

    return corr_id

# ---------- Internals ----------
def _push_fcm_v1(user_id: int, title: str, body: str, data: dict) -> None:
    project_id = getattr(settings, "FCM_PROJECT_ID", None) or _get_project_id_from_sa()
    if not project_id:
        logger.warning("FCM: project id missing; set FCM_PROJECT_ID or FIREBASE_SERVICE_ACCOUNT_B64")
        return

    # ✅ ONLY mobile tokens (prevents "web:..." 400 INVALID_ARGUMENT)
    qs = (
        DeviceToken.objects
        .filter(user_id=user_id, is_active=True, platform__in=["android", "ios"])
        .order_by("-last_seen", "-created_at")
        .values_list("token", flat=True)
    )

    # ✅ dedupe tokens
    seen = set()
    tokens = []
    for t in qs:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)

    if not tokens:
        logger.debug("FCM: no active mobile tokens for user %s", user_id)
        return

    access_token = _get_google_oauth_token()
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    notif = {"title": title, "body": body}
    data  = {str(k): str(v) for k, v in (data or {}).items()}

    for t in tokens:
        msg = {
            "message": {
                "token": t,
                "notification": notif,
                "data": data,
                "android": {"priority": "HIGH"},
                "apns": {"headers": {"apns-priority": "10"}, "payload": {"aps": {"sound": "default"}}},
            }
        }

        resp = requests.post(url, headers=headers, data=json.dumps(msg), timeout=10)

        if resp.ok:
            logger.info("FCM v1 OK user=%s token=%s… resp=%s", user_id, t[:12], resp.text.strip())
            continue

        if resp.status_code in (404, 410) or _response_has_fcm_error(resp, "UNREGISTERED"):
            deleted, _ = DeviceToken.objects.filter(token=t).delete()
            logger.info("FCM: deleted UNREGISTERED token user=%s token=%s… rows=%s", user_id, t[:12], deleted)
            continue

        logger.warning(
            "FCM v1 ERR user=%s token=%s… code=%s body=%s",
            user_id, t[:12], resp.status_code, resp.text.strip()
        )


def _response_has_fcm_error(resp, code: str) -> bool:
    try:
        j = resp.json()
        details = (j.get("error", {}) or {}).get("details", []) or []
        for d in details:
            if isinstance(d, dict) and d.get("errorCode") == code:
                return True
    except Exception:
        pass
    return False

def _maybe_deactivate_on_unregistered(token: str, resp) -> None:
    try:
        j = resp.json()
        details = j.get("error", {}).get("details", [])
        if any(isinstance(d, dict) and d.get("errorCode") == "UNREGISTERED" for d in details):
            DeviceToken.objects.filter(token=token).update(is_active=False)
            logger.info("Deactivated UNREGISTERED token")
    except Exception:
        pass

def _get_google_oauth_token() -> str:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    scopes = ["https://www.googleapis.com/auth/firebase.messaging"]
    b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64")
    if b64:
        info = json.loads(base64.b64decode(b64).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    else:
        path = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", None)
        if not path:
            raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_B64 or GOOGLE_APPLICATION_CREDENTIALS")
        creds = service_account.Credentials.from_service_account_file(path, scopes=scopes)
    creds.refresh(Request())
    return creds.token

def _get_project_id_from_sa() -> Optional[str]:
    try:
        b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64")
        if b64:
            return json.loads(base64.b64decode(b64).decode("utf-8")).get("project_id")
        path = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", None)
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("project_id")
    except Exception:
        pass
    return None





from mobicrowd.models.notifications import Notification
from mobicrowd.emoji_codec import encode_for_db, decode_for_api

# uses existing _push_fcm_v1(...) already in this file
# uses existing logger already in this file


class NotifyItem(TypedDict):
    user_id: int
    title: str
    body: str
    payload: dict
    priority: str

from django.utils import timezone as dj_tz
def notify_users_bulk(
    *,
    event_type: str,
    items: Sequence[NotifyItem],
    match_filters: dict | None = None) -> None:
    """
    Bulk persist notifications, then fan out WS + FCM per user.
    match_filters is used to re-select the exact inserted rows (e.g. payload__event_id=...).
    """
    if not items:
        return

    match_filters = match_filters or {}
    now = dj_tz.now()

    # 1) Persist (single DB op)
    objs = [
        Notification(
            user_id=i["user_id"],
            event_type=event_type,
            title=encode_for_db(i["title"]),
            body=encode_for_db(i.get("body", "")),
            payload=i.get("payload") or {},
            priority=i.get("priority", "normal"),
            created_at=now,  # ensures value even if DB/engine doesn't apply auto_now_add in bulk
        )
        for i in items
    ]
    Notification.objects.bulk_create(objs, batch_size=500)

    # 2) Re-select inserted rows to get ids/created_at (needed by WS + FCM data.notification_id)
    user_ids = [i["user_id"] for i in items]
    qs = (
        Notification.objects.filter(
            user_id__in=user_ids,
            event_type=event_type,
            created_at__gte=now - timedelta(seconds=5),
            **match_filters,
        )
        .order_by("id")
    )

    # If duplicates exist for any reason, keep the latest per user
    latest_by_user: dict[int, Notification] = {}
    for n in qs:
        latest_by_user[n.user_id] = n

    channel_layer = get_channel_layer()

    # 3) Fanout WS + FCM per user (messages are personalized already)
    for uid, n in latest_by_user.items():
        payload = n.payload or {}
        priority = n.priority

        # WS
        try:
            async_to_sync(channel_layer.group_send)(
                f"user_{uid}",
                {
                    "type": "notify",
                    "message": {
                        "kind": "new_notification",
                        "notification": {
                            "id": n.id,
                            "event_type": event_type,
                            "title": decode_for_api(n.title),
                            "body": decode_for_api(n.body),
                            "payload": payload,
                            "priority": priority,
                            "created_at": n.created_at.isoformat(),
                        },
                    },
                },
            )
        except Exception as e:
            logger.warning("WS fanout failed for user %s: %s", uid, e)

        # FCM
        try:
            _push_fcm_v1(
                uid,
                decode_for_api(n.title),
                decode_for_api(n.body),
                data={"notification_id": n.id, **payload},
            )
        except Exception as e:
            logger.error("FCM push failed for user %s: %s", uid, e)
