# mobicrowd/views/apis/submission_flow/push_silent.py
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Iterable

import requests
from django.conf import settings
from django.db.models import QuerySet

from mobicrowd.models.notifications import DeviceToken

logger = logging.getLogger("mobicrowd.push_silent")


# ---------------------------
# FCM v1 (no firebase_admin)
# ---------------------------

def _stringify_data(d: Dict[str, Any]) -> Dict[str, str]:
    """
    FCM 'data' values MUST be strings.
    - dict/list are json-dumped
    - None is omitted
    """
    out: Dict[str, str] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _load_service_account_json() -> Optional[Dict[str, Any]]:
    """
    Mirrors your notify.py behavior:
    - Prefer env FIREBASE_SERVICE_ACCOUNT_B64 (base64 JSON)
    - Else allow GOOGLE_APPLICATION_CREDENTIALS path
    """
    b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_B64")
    if b64:
        try:
            raw = base64.b64decode(b64).decode("utf-8")
            return json.loads(raw)
        except Exception as e:
            logger.error("Invalid FIREBASE_SERVICE_ACCOUNT_B64: %s", e)
            return None

    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed reading GOOGLE_APPLICATION_CREDENTIALS JSON: %s", e)
            return None

    return None


def _get_project_id_from_sa(sa: Dict[str, Any]) -> Optional[str]:
    return sa.get("project_id") if isinstance(sa, dict) else None


def _get_google_oauth_token(sa: Dict[str, Any]) -> Optional[str]:
    """
    Create OAuth2 token for FCM v1 using service account.
    Requires google-auth (already needed by your notify.py).
    """
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GoogleRequest
    except Exception as e:
        logger.error("Missing google-auth dependency: %s", e)
        return None

    try:
        scopes = ["https://www.googleapis.com/auth/firebase.messaging"]
        creds = service_account.Credentials.from_service_account_info(sa, scopes=scopes)
        creds.refresh(GoogleRequest())
        return creds.token
    except Exception as e:
        logger.error("Failed to mint OAuth token for FCM v1: %s", e)
        return None


def _is_unregistered_fcm(resp_json: Dict[str, Any]) -> bool:
    """
    FCM v1 unregistered token typically appears in:
      error.details[].errorCode == "UNREGISTERED"
    """
    try:
        err = resp_json.get("error") or {}
        details = err.get("details") or []
        for d in details:
            if isinstance(d, dict) and d.get("errorCode") == "UNREGISTERED":
                return True
        return False
    except Exception:
        return False


def _deactivate_token(token: str) -> None:
    try:
        DeviceToken.objects.filter(token=token).update(is_active=False)
    except Exception:
        pass


def _iter_user_tokens(user_id: int) -> Iterable[Tuple[str, str]]:
    """
    Returns (token, platform) for active tokens.
    platform: 'ios' | 'android' | 'web' (per your model)
    """
    qs = (DeviceToken.objects
          .filter(user_id=user_id, is_active=True)
          .values_list("token", "platform")
          .distinct())
    return list(qs)


def push_data_only(
    *,
    user_id: int,
    event_type: str,
    payload: Dict[str, Any],
    priority: str = "high",
    collapse_id: Optional[str] = None,
) -> None:
    """
    Sends a silent/data-only push (NO Notification row).
    Mobile app handles it in background:
      - event_type: e.g. "submission.original.request"
      - payload: dict (stringified / json-dumped for FCM data)

    Notes:
      - Data payload is capped (~4KB). Keep payload compact.
      - For iOS silent delivery: aps.content-available=1 + apns-push-type=background
      - For Android: android.priority HIGH helps delivery
    """
    tokens = list(_iter_user_tokens(user_id))
    if not tokens:
        return

    # Determine project/service account
    sa = _load_service_account_json()
    if not sa:
        logger.error("No service account configured (FIREBASE_SERVICE_ACCOUNT_B64 or GOOGLE_APPLICATION_CREDENTIALS).")
        return

    project_id = getattr(settings, "FCM_PROJECT_ID", None) or _get_project_id_from_sa(sa)
    if not project_id:
        logger.error("FCM project_id missing (set settings.FCM_PROJECT_ID or include it in service account JSON).")
        return

    access_token = _get_google_oauth_token(sa)
    if not access_token:
        return

    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    # FCM data message
    data = _stringify_data({
        "event_type": event_type,
        "payload": payload,  # will be json-dumped to string
    })

    android_priority = "HIGH" if str(priority).lower() == "high" else "NORMAL"
    apns_priority = "10" if str(priority).lower() == "high" else "5"

    for token, platform in tokens:
        # Build per-platform configs to keep semantics correct.
        message: Dict[str, Any] = {
            "token": token,
            "data": data,
        }

        if platform == "android":
            message["android"] = {
                "priority": android_priority,
                **({"collapse_key": collapse_id} if collapse_id else {}),
            }

        elif platform == "ios":
            # iOS silent: background push type + content-available
            apns_headers = {
                "apns-push-type": "background",
                "apns-priority": apns_priority,
            }
            if collapse_id:
                apns_headers["apns-collapse-id"] = collapse_id

            message["apns"] = {
                "headers": apns_headers,
                "payload": {
                    "aps": {
                        "content-available": 1
                    }
                },
            }

        else:
            # web/unknown: still send data-only; some clients may ignore it.
            # If you later need WebPush, add message["webpush"] here.
            pass

        body = {"message": message}

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10)
        except Exception as e:
            logger.warning("FCM request failed (token=%s): %s", token[:12], e)
            continue

        if resp.status_code == 200:
            continue

        # Handle invalid/unregistered token
        try:
            j = resp.json()
        except Exception:
            j = {}

        if resp.status_code in (400, 401, 403, 404, 410) and _is_unregistered_fcm(j):
            _deactivate_token(token)
            continue

        # Log other errors (do not crash celery)
        logger.warning(
            "FCM send failed status=%s token=%s resp=%s",
            resp.status_code,
            token[:12],
            (resp.text[:400] if hasattr(resp, "text") else str(j)),
        )
