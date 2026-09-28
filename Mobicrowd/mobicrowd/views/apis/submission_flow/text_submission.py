from __future__ import annotations

import logging
from typing import Any, Dict

from django.db import transaction
from django.utils import timezone

from mobicrowd.models.submisson import Event, Submission
from mobicrowd.views.apis.submission_flow.security import (
    approved_capacity_available,
    mark_refused,
    text_digest,
)
from mobicrowd.views.apis.textual_pipeline.helpers import (
    check_text_redundancy,
    check_text_safety,
    check_textual_relevance,
    delete_text_embedding,
)

logger = logging.getLogger("mobicrowd.submission_flow.text")


class TextPipelineTechnicalError(RuntimeError):
    pass


def _refuse(submission_id: int, message: str) -> Submission:
    with transaction.atomic():
        submission = Submission.objects.select_for_update().get(pk=submission_id)
        if submission.status == Submission.APPROVED:
            raise TextPipelineTechnicalError("Approved text submission cannot be re-refused.")
        mark_refused(submission=submission, message=message)
        return submission


def validate_and_finalize_text_submission(submission_id: int) -> Dict[str, Any]:
    """
    Authoritative text pipeline.

    The client cannot provide validation status, requester text, similarity decision,
    or approval. Relevance is evaluated against the server-side Event.description.
    The event row is locked across the authoritative redundancy decision + final DB
    transition so concurrent text submissions for the same event cannot both pass
    the same duplicate/capacity boundary.
    """
    submission = (
        Submission.objects.select_related("event", "worker", "worker__user")
        .get(pk=submission_id)
    )
    if submission.status != Submission.PENDING:
        raise TextPipelineTechnicalError(f"Unexpected submission status: {submission.status}")
    if submission.flow_stage != Submission.FLOW_VALIDATING:
        raise TextPipelineTechnicalError(f"Unexpected flow stage: {submission.flow_stage}")
    if submission.photo_id or submission.video_id or not (submission.text or "").strip():
        raise TextPipelineTechnicalError("Submission is not a text-only submission.")

    provider_text = (submission.text or "").strip()
    requester_text = (submission.event.description or "").strip()
    expected_digest = submission.flow_artifact_digest
    if not expected_digest or text_digest(provider_text) != expected_digest:
        _refuse(submission_id, "Text submission rejected: content changed during validation.")
        return {"ok": False, "decision": "REJECT", "stage": "integrity", "reason": "text_changed"}

    # Safety failure caused by provider content is a rejection. A provider/LLM
    # infrastructure failure is a technical error and must not be converted into a
    # contributor rejection or approval.
    safety = check_text_safety(provider_text, use_llm=True, output_language="auto")
    if not safety.get("ok") or safety.get("llm_error"):
        raise TextPipelineTechnicalError(
            safety.get("error") or safety.get("llm_error") or "Text safety validation failed."
        )
    if not bool(safety.get("is_safe_for_work", False)):
        reason = str(safety.get("reason") or "Text submission rejected by safety validation.")
        _refuse(submission_id, reason)
        return {"ok": False, "decision": "REJECT", "stage": "safety", "reason": reason}

    relevance = check_textual_relevance(
        requester_text=requester_text,
        provider_text=provider_text,
        output_language="auto",
    )
    if not relevance.get("ok"):
        raise TextPipelineTechnicalError(relevance.get("error") or "Text relevance validation failed.")
    if not bool(relevance.get("is_relevant", False)):
        reason = str(relevance.get("reason") or "Text submission is not relevant to the event request.")
        _refuse(submission_id, reason)
        return {
            "ok": False,
            "decision": "REJECT",
            "stage": "relevance",
            "reason": reason,
            "score": relevance.get("score"),
        }

    stored_id = None
    try:
        # Serialize the authoritative duplicate/capacity/finalization boundary per
        # event. This prevents two simultaneous accepted-text attempts from both
        # observing the pre-insert Qdrant state and both becoming APPROVED.
        with transaction.atomic():
            locked = (
                Submission.objects.select_for_update()
                .select_related("event", "worker", "worker__user")
                .get(pk=submission_id)
            )
            locked.event = Event.objects.select_for_update().get(pk=locked.event_id)

            if locked.status != Submission.PENDING or locked.flow_stage != Submission.FLOW_VALIDATING:
                raise TextPipelineTechnicalError("Text workflow state changed before approval.")
            if text_digest(locked.text or "") != expected_digest:
                mark_refused(
                    submission=locked,
                    message="Text submission rejected: content changed during validation.",
                )
                return {"ok": False, "decision": "REJECT", "stage": "integrity"}

            capacity_ok, capacity_reason = approved_capacity_available(locked)
            if not capacity_ok:
                mark_refused(
                    submission=locked,
                    message="Text submission rejected: event submission capacity reached.",
                )
                return {
                    "ok": False,
                    "decision": "REJECT",
                    "stage": "capacity",
                    "reason": capacity_reason,
                }

            redundancy = check_text_redundancy(
                provider_text=provider_text,
                requester_text=requester_text,
                file_id=str(locked.id),
                event_id=int(locked.event_id),
                persist=True,
            )
            if not redundancy.get("ok"):
                raise TextPipelineTechnicalError(
                    redundancy.get("error") or "Text redundancy validation failed."
                )
            if bool(redundancy.get("is_duplicate", False)):
                reason = "Text submission rejected due to redundancy."
                mark_refused(submission=locked, message=reason)
                return {
                    "ok": False,
                    "decision": "REJECT",
                    "stage": "redundancy",
                    "reason": redundancy.get("reason"),
                    "similarity": redundancy.get("similarity"),
                }

            stored_id = redundancy.get("stored_id")

            locked.status = Submission.APPROVED
            locked.flow_stage = Submission.FLOW_FINALIZED
            locked.flow_validated_at = timezone.now()
            locked.flow_finalized_at = locked.flow_validated_at
            locked.message = "Submission accepted"
            locked.save(
                update_fields=[
                    "status",
                    "flow_stage",
                    "flow_validated_at",
                    "flow_finalized_at",
                    "message",
                ]
            )
    except Exception:
        # If Qdrant persistence succeeded but the DB transaction could not commit,
        # compensate so a failed attempt cannot poison future duplicate checks.
        if stored_id:
            try:
                delete_text_embedding(str(stored_id))
            except Exception:
                logger.exception("Failed to compensate text Qdrant point %s", stored_id)
        raise

    return {
        "ok": True,
        "decision": "ACCEPT",
        "stage": "finalized",
        "relevance_score": relevance.get("score"),
        "redundancy_similarity": redundancy.get("similarity"),
    }
