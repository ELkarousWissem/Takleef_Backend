import logging
from typing import Mapping, Optional, Any, Dict

from django.db import transaction

from mobicrowd.models.submisson import Submission
from mobicrowd.notify import notify_user

logger = logging.getLogger("mobicrowd.submission_flow")
DECISION_EVENT_TYPE = "submission.decision"


def refuse_and_notify(
    *,
    submission_id: int,
    user_id: int,
    reason: str,
    message_db: str,
    title: str = "Submission rejected",
    body: str = "Your submission was rejected.",
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Server-side final rejection. Synchronizes public and internal workflow state."""
    with transaction.atomic():
        submission = (
            Submission.objects.select_for_update()
            .select_related("worker", "worker__user")
            .get(id=submission_id)
        )
        owner_id = int(submission.worker.user_id)

        if submission.status == Submission.APPROVED:
            # Never silently turn a finalized approved submission into refused from a stale task.
            return {
                "stop": True,
                "submission_id": submission_id,
                "user_id": owner_id,
                "reason": "already_approved",
            }
        if submission.status == Submission.REFUSED:
            if submission.flow_stage != Submission.FLOW_REFUSED:
                submission.flow_stage = Submission.FLOW_REFUSED
                submission.save(update_fields=["flow_stage"])
            return {
                "stop": True,
                "submission_id": submission_id,
                "user_id": owner_id,
                "reason": "already_refused",
            }

        submission.status = Submission.REFUSED
        submission.flow_stage = Submission.FLOW_REFUSED
        submission.message = message_db
        submission.save(update_fields=["status", "flow_stage", "message"])

    payload = {
        "submission_id": submission_id,
        "decision": "REJECT",
        "reason": reason,
    }
    if extra_payload:
        payload.update(dict(extra_payload))

    try:
        notify_user(
            user_id=owner_id,
            event_type=DECISION_EVENT_TYPE,
            title=title,
            body=body,
            payload=payload,
            priority="high",
        )
    except Exception:
        logger.exception("Failed to send rejection push (submission_id=%s)", submission_id)

    return {
        "stop": True,
        "submission_id": submission_id,
        "user_id": owner_id,
        "reason": reason,
    }
