from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Optional

from django.conf import settings
from django.core import signing
from django.core.exceptions import ImproperlyConfigured
from django.utils import timezone
from django.db.models import Q
from rest_framework.exceptions import PermissionDenied

from mobicrowd.models.submisson import EventWorker, Submission


FINALIZE_TOKEN_SALT = "mobicrowd.submission.finalize.v3"
FINALIZE_TOKEN_VERSION = 3


class FlowSecurityError(Exception):
    """Permanent workflow/security violation. Do not retry as a normal transient failure."""


class FlowConfigurationError(FlowSecurityError):
    """Required production security setting is missing or invalid."""


def submission_media_type(submission: Submission) -> str:
    has_photo = bool(getattr(submission, "photo_id", None))
    has_video = bool(getattr(submission, "video_id", None))
    has_text = bool((getattr(submission, "text", None) or "").strip())

    present = [
        name
        for name, enabled in (("photo", has_photo), ("video", has_video), ("text", has_text))
        if enabled
    ]
    if len(present) != 1:
        raise FlowSecurityError(
            f"Submission must contain exactly one media type; found {present or 'none'}."
        )
    return present[0]


def assert_submission_owner(submission: Submission, user) -> None:
    worker = getattr(submission, "worker", None)
    if worker is None or worker.user_id != user.id:
        raise PermissionDenied("This submission does not belong to the authenticated contributor.")


def assert_worker_event_approved(*, submission: Submission) -> None:
    if not EventWorker.objects.filter(
        event_id=submission.event_id,
        worker_id=submission.worker_id,
        status=EventWorker.APPROVED,
    ).exists():
        raise FlowSecurityError("Contributor is not approved for this event.")


def assert_event_open(submission: Submission) -> None:
    now = timezone.now()
    event = submission.event
    if event.startdate and now < event.startdate:
        raise FlowSecurityError("Event has not started yet.")
    if event.deadline and now > event.deadline:
        raise FlowSecurityError("Event has ended.")


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FlowSecurityError("Decoded artifact is missing.")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FlowSecurityError("Decoded video-frame directory is missing.")
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise FlowSecurityError("Decoded video-frame directory is empty.")

    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def text_digest(text: str) -> str:
    normalized = " ".join((text or "").strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_ttl_seconds() -> int:
    raw = getattr(settings, "SUBMISSION_FINALIZE_TOKEN_TTL_SECONDS", 900)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(
            "SUBMISSION_FINALIZE_TOKEN_TTL_SECONDS must be a positive integer."
        ) from exc
    if value <= 0:
        raise ImproperlyConfigured(
            "SUBMISSION_FINALIZE_TOKEN_TTL_SECONDS must be a positive integer."
        )
    return value


def issue_finalize_token(submission: Submission) -> str:
    if submission.flow_stage != Submission.FLOW_VALIDATED:
        raise FlowSecurityError("Submission has not passed validation.")
    if not submission.flow_artifact_digest:
        raise FlowSecurityError("Validated artifact digest is missing.")

    payload = {
        "v": FINALIZE_TOKEN_VERSION,
        "sid": int(submission.id),
        "uid": int(submission.worker.user_id),
        "nonce": str(submission.flow_nonce),
        "media": submission_media_type(submission),
        "digest": str(submission.flow_artifact_digest),
    }
    return signing.dumps(payload, salt=FINALIZE_TOKEN_SALT, compress=True)


def load_finalize_token(token: str) -> dict:
    if not token:
        raise signing.BadSignature("Missing token")
    return signing.loads(
        token,
        salt=FINALIZE_TOKEN_SALT,
        max_age=_token_ttl_seconds(),
    )


def assert_finalize_token_matches(
    *,
    payload: dict,
    submission: Submission,
    request_user,
    expected_media_type: str,
) -> None:
    if int(payload.get("v", -1)) != FINALIZE_TOKEN_VERSION:
        raise FlowSecurityError("Unsupported finalize-token version.")

    expected = {
        "sid": int(submission.id),
        "uid": int(request_user.id),
        "nonce": str(submission.flow_nonce),
        "media": expected_media_type,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FlowSecurityError(f"Finalize token does not match submission ({key}).")

    token_digest = str(payload.get("digest") or "")
    stored_digest = str(submission.flow_artifact_digest or "")
    if not token_digest or not stored_digest or not hmac.compare_digest(token_digest, stored_digest):
        raise FlowSecurityError("Finalize token does not match validated content.")


def configured_binding_threshold(setting_name: str) -> float:
    """
    Original-vs-validation-proxy similarity threshold.

    This intentionally fails closed when absent. The correct threshold depends on the
    deployed encoder/reconstruction pipeline and must be calibrated on genuine
    matched/unmatched samples rather than guessed in application code.
    """
    raw = getattr(settings, setting_name, None)
    if raw is None:
        raise FlowConfigurationError(
            f"{setting_name} is required. Calibrate it before enabling production finalization."
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise FlowConfigurationError(f"{setting_name} must be a float in [-1, 1].") from exc
    if not -1.0 <= value <= 1.0:
        raise FlowConfigurationError(f"{setting_name} must be a float in [-1, 1].")
    return value


def sanitize_filename(name: str) -> str:
    raw = str(name or "").strip()
    clean = Path(raw).name
    if not raw or clean != raw or clean in {".", ".."}:
        raise FlowSecurityError("Invalid file name.")
    return clean


def mark_refused(*, submission: Submission, message: str) -> None:
    submission.status = Submission.REFUSED
    submission.flow_stage = Submission.FLOW_REFUSED
    submission.message = message
    submission.save(update_fields=["status", "flow_stage", "message"])


def approved_capacity_available(submission: Submission) -> tuple[bool, Optional[str]]:
    """
    Must be called while the event row is locked by the caller when race-free capacity
    enforcement is required. Refused attempts do not consume quota.
    """
    media = submission_media_type(submission)
    event = submission.event

    # FINALIZING rows are reservations. Counting them prevents two concurrent
    # valid tokens from racing past the same remaining quota slot.
    qs = Submission.objects.filter(event_id=event.id).exclude(pk=submission.pk).filter(
        Q(status=Submission.APPROVED) | Q(flow_stage=Submission.FLOW_FINALIZING)
    )

    if media == "photo":
        qs = qs.filter(photo__isnull=False)
        per_worker = int(getattr(event, "max_photos_per_worker", 0) or 0)
        event_limit = int(getattr(event, "numberOfPhotos", 0) or 0)
    elif media == "video":
        qs = qs.filter(video__isnull=False)
        per_worker = int(getattr(event, "max_videos_per_worker", 0) or 0)
        event_limit = int(getattr(event, "numberOfVideos", 0) or 0)
    else:
        qs = qs.filter(photo__isnull=True, video__isnull=True, text__isnull=False).exclude(text="")
        per_worker = int(getattr(event, "max_texts_per_worker", 0) or 0)
        event_limit = int(getattr(event, "numberOfTexts", 0) or 0)

    if per_worker > 0 and qs.filter(worker_id=submission.worker_id).count() >= per_worker:
        return False, f"{media}_worker_quota_reached"
    if event_limit > 0 and qs.count() >= event_limit:
        return False, f"{media}_event_quota_reached"
    return True, None
