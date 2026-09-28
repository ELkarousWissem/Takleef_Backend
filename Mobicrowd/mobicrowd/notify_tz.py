# mobicrowd/notify_tz.py
from __future__ import annotations

from datetime import datetime, timezone as dt_timezone  # stdlib timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

from django.utils import timezone as dj_tz  # Django timezone utilities

from mobicrowd.models.notifications import DeviceToken  # adjust path if needed

# ✅ tzinfo *instance* for UTC (required by make_aware/astimezone)
UTC = dt_timezone.utc

def get_user_tzname(user_id: int) -> Optional[str]:
    tok = (
        DeviceToken.objects.filter(user_id=user_id, is_active=True)
        .order_by("-last_seen")
        .only("timezone")
        .first()
    )
    return tok.timezone if tok and tok.timezone else None

def fmt_for_user(dt: datetime, user_id: int, fmt: str = "%b %d, %H:%M %Z") -> str:
    tzname = get_user_tzname(user_id)
    try:
        tzinfo = ZoneInfo(tzname) if tzname else dj_tz.get_current_timezone()
    except Exception:
        tzinfo = dj_tz.get_current_timezone()
    return dj_tz.localtime(dt, tzinfo).strftime(fmt)

def iso_utc(dt: datetime) -> str:
    """
    Return a strict UTC ISO-8601 string with trailing 'Z'.
    Accepts aware or naive datetimes; naive treated as UTC.
    """
    if dj_tz.is_naive(dt):
        dt = dj_tz.make_aware(dt, UTC)   # tzinfo instance
    dt = dt.astimezone(UTC)              # tzinfo instance
    return dt.isoformat().replace("+00:00", "Z")

def tz_map_for_users(user_ids: List[int]) -> Dict[int, str]:
    rows = (
        DeviceToken.objects.filter(user_id__in=user_ids, is_active=True)
        .order_by("user_id", "-last_seen")
        .values("user_id", "timezone")
    )
    out: Dict[int, str] = {}
    for r in rows:
        uid = r["user_id"]
        if uid not in out and r["timezone"]:
            out[uid] = r["timezone"]
    return out

def fmt_with_map(dt: datetime, user_id: int, tzmap: Dict[int, str],
                 fmt: str = "%b %d, %H:%M %Z") -> str:
    tzname = tzmap.get(user_id)
    try:
        tzinfo = ZoneInfo(tzname) if tzname else dj_tz.get_current_timezone()
    except Exception:
        tzinfo = dj_tz.get_current_timezone()
    return dj_tz.localtime(dt, tzinfo).strftime(fmt)
