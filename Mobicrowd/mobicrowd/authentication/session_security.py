import hashlib
import random
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from mobicrowd.models.notifications import DeviceToken, LoginTakeoverChallenge


def session_timeout_for_kind(kind: str):
    """Return the inactivity timeout, or None for a persistent session."""
    if kind == "DASHBOARD":
        minutes = getattr(settings, "DASHBOARD_SESSION_TIMEOUT_MINUTES", None)
    else:
        minutes = getattr(settings, "APP_SESSION_TIMEOUT_MINUTES", None)

    if minutes is None:
        return None

    minutes = int(minutes)
    if minutes <= 0:
        return None

    return timedelta(minutes=minutes)


def is_session_stale(row: DeviceToken, now=None) -> bool:
    """Persistent sessions never become stale only because of inactivity."""
    timeout = session_timeout_for_kind(row.kind)

    if timeout is None:
        return False

    now = now or timezone.now()

    if not row.session_last_seen:
        return True

    return row.session_last_seen < now - timeout


def deactivate_stale_sessions(user, kind: str) -> int:
    """Release stale sessions only when an inactivity timeout is configured."""
    timeout = session_timeout_for_kind(kind)

    if timeout is None:
        return 0

    now = timezone.now()
    stale_before = now - timeout

    stale_qs = DeviceToken.objects.filter(
        user=user,
        kind=kind,
        session_active=True,
        session_last_seen__lt=stale_before,
    )

    null_seen_qs = DeviceToken.objects.filter(
        user=user,
        kind=kind,
        session_active=True,
        session_last_seen__isnull=True,
    )

    count = stale_qs.update(
        session_active=False,
        session_sid="",
        is_active=False,
    )

    count += null_seen_qs.update(
        session_active=False,
        session_sid="",
        is_active=False,
    )

    return count

def deactivate_other_sessions(user, kind: str, keep_device_uid: str):
    return DeviceToken.objects.filter(
        user=user,
        kind=kind,
        session_active=True,
    ).exclude(
        device_uid=keep_device_uid,
    ).update(
        session_active=False,
        session_sid="",
        is_active=False,
    )


def hash_otp(raw_otp: str) -> str:
    secret = getattr(settings, "SECRET_KEY", "")
    value = f"{raw_otp}:{secret}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"


def expire_old_challenges(user, kind: str, incoming_device_uid: str):
    LoginTakeoverChallenge.objects.filter(
        user=user,
        kind=kind,
        incoming_device_uid=incoming_device_uid,
        status=LoginTakeoverChallenge.STATUS_PENDING,
    ).update(
        status=LoginTakeoverChallenge.STATUS_EXPIRED,
        decided_at=timezone.now(),
    )


@transaction.atomic
def create_email_otp_challenge(*, user, kind, platform, incoming_device_uid, incoming_device_label=""):
    expire_old_challenges(user, kind, incoming_device_uid)

    raw_otp = generate_otp()

    challenge = LoginTakeoverChallenge.objects.create(
        user=user,
        method=LoginTakeoverChallenge.METHOD_EMAIL_OTP,
        kind=kind,
        platform=platform,
        incoming_device_uid=incoming_device_uid,
        incoming_device_label=incoming_device_label or "",
        otp_hash=hash_otp(raw_otp),
        expires_at=LoginTakeoverChallenge.new_expiry(),
    )

    return challenge, raw_otp


@transaction.atomic
def create_active_session_challenge(*, user, kind, platform, incoming_device_uid, incoming_device_label=""):
    expire_old_challenges(user, kind, incoming_device_uid)

    challenge = LoginTakeoverChallenge.objects.create(
        user=user,
        method=LoginTakeoverChallenge.METHOD_ACTIVE_SESSION,
        kind=kind,
        platform=platform,
        incoming_device_uid=incoming_device_uid,
        incoming_device_label=incoming_device_label or "",
        expires_at=LoginTakeoverChallenge.new_expiry(),
    )

    return challenge