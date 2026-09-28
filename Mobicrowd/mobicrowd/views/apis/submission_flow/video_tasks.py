import os
import time

import torch
import cv2
import numpy as np
from PIL import Image
from celery import shared_task
from qdrant_client import QdrantClient

from django.conf import settings
from django.db import transaction
from django.utils import timezone as dj_timezone
from mobicrowd.models.submisson import Video, Submission, UserUploadLog, Event
from mobicrowd.tasks import clip_model, clip_processor, device
from mobicrowd.video_tasks import blur_video_faces_sparse
from mobicrowd.views.apis.submission_flow.qwen3_video_relevance import \
    openrouter_qwen3_vl_multiframe_caption_and_relevance
import io
import base64
import tempfile
import subprocess
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any

from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.files.base import ContentFile
from PIL import Image, ImageOps

from mobicrowd.models.submisson import Submission
from mobicrowd.models.FileStorage import CustomS3Boto3Storage
from mobicrowd.notify import signal_user
from mobicrowd.views.apis.submission_flow.security import (
    approved_capacity_available,
    configured_binding_threshold,
    mark_refused,
)

import torch
import torch.nn.functional as F
logger = get_task_logger("mobicrowd.submission_flow.video")
from pathlib import Path
from django.conf import settings

BASE_DIR = Path(__file__).resolve().parent.parent
SAFE_TMP_ROOT = (BASE_DIR / "tmp").resolve()

def _extract_feature_tensor(outputs):
    """
    Supports different CLIP/vision model output formats:
    - Tensor
    - outputs.image_embeds
    - outputs.pooler_output
    - outputs.last_hidden_state
    - tuple/list tensor fallback
    """
    if isinstance(outputs, torch.Tensor):
        return outputs

    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state.mean(dim=1)

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if isinstance(item, torch.Tensor):
                return item

    raise TypeError(f"Unsupported CLIP output type: {type(outputs)}")


def clip_video_embedding_from_frames_dir(frames_dir: str) -> list[float]:
    frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    if len(frames) != 8:
        raise ValueError(f"Expected exactly 8 frames, found {len(frames)} in {frames_dir}")

    clip_model.eval()
    embs = []

    with torch.no_grad():
        for p in frames:
            img = Image.open(p).convert("RGB")

            inputs = clip_processor(
                text="prompts",
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=clip_processor.tokenizer.model_max_length,
            ).to(device)

            # Use same safe pattern as your photo embedding path
            outputs = clip_model(**inputs)
            feats = _extract_feature_tensor(outputs)

            feats = F.normalize(feats, p=2, dim=-1)
            embs.append(feats[0].detach().cpu())

    mean_emb = torch.stack(embs, dim=0).mean(dim=0)
    mean_emb = F.normalize(mean_emb, p=2, dim=0)

    return mean_emb.tolist()

from pathlib import Path
def _safe_delete_dir(p: Path) -> None:
    """
    Delete a directory only if it’s under SAFE_TMP_ROOT.
    Prevents deleting anything outside BASE_DIR/tmp.
    """
    p = p.resolve()
    try:
        p.relative_to(SAFE_TMP_ROOT)
    except ValueError:
        logger.warning("Skip deletion outside safe root: %s", p)
        return

    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def stage_relevance_video(
    self,
    *,
    submission_id: int,
    user_id: int,
    frames_dir: str,
    description: str
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    submission = Submission.objects.select_related("video", "event", "worker", "worker__user").get(pk=submission_id)
    if submission.worker.user_id != int(user_id):
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "owner_mismatch"}
    if submission.flow_stage != Submission.FLOW_VALIDATING:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "invalid_flow_state"}
    # idempotency
    if submission.status in (Submission.APPROVED, Submission.REFUSED):
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "already_final"}


    # 1) Multi-frame VLM relevance + caption
    keys_file = settings.OPENROUTER_KEYS_FILE
    model_id = settings.OPENROUTER_VIDEO_RELEVANCE_MODEL
    fallback_models = list(
        getattr(settings, "OPENROUTER_VIDEO_RELEVANCE_FALLBACK_MODELS", [])
    )

    if not keys_file or not os.path.exists(keys_file):
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {keys_file}")
    if not model_id:
        raise RuntimeError("OPENROUTER_VIDEO_RELEVANCE_MODEL is not configured.")

    res = openrouter_qwen3_vl_multiframe_caption_and_relevance(
        frames_dir=frames_dir,
        description=(description or "").strip(),
        keys_file=keys_file,
        logger=logger,
        model_id=model_id,
        fallback_models=fallback_models,
        expected_frames=8,
        max_attempts=10,
        max_tokens=220,
        jpeg_quality=75,
        timeout_s=180,
    )

    if not res.get("ok"):
        raise RuntimeError(
            "OpenRouter video relevance failed after configured model fallbacks: "
            f"{res.get('reason')} / err={res.get('last_error')}"
        )

    # Canonical recapture flag propagated by the multi-frame VLM helper.
    # A confirmed recapture is always treated as not relevant.
    recapture = bool(res.get("recapture", False))
    is_relevant = (res.get("relevant") == "YES") and not recapture

    analysis = {
        "is_relevant": bool(is_relevant),
        "recapture": recapture,
        "vlm_model": res.get("model"),
        "vlm_requested_model": model_id,
        "vlm_configured_fallback_models": fallback_models,
        "vlm_model_fallback_used": bool(res.get("model") and res.get("model") != model_id),
        "vlm_attempt": res.get("attempt"),
        "vlm_key_used": res.get("key_used"),
        "vlm_latency_s": float(res.get("latency_seconds") or 0.0),
        "vlm_cost": res.get("cost"),
        "vlm_usage": res.get("usage") or {},
        "vlm_reason": (res.get("reason") or "")[:300],
        "vlm_caption": (res.get("caption") or "")[:600],
        "vlm_unwanted_content": res.get("unwanted_content", "NO"),
        "vlm_unwanted_categories": res.get("unwanted_categories") or [],
        "vlm_openrouter_id": res.get("openrouter_id"),
    }

    logger.info(
        "[stage_relevance_video] VLM_DECISION submission_id=%s relevant=%s recapture=%s "
        "unwanted_content=%s unwanted_categories=%s reason=%r",
        submission_id,
        is_relevant,
        recapture,
        analysis.get("vlm_unwanted_content"),
        analysis.get("vlm_unwanted_categories"),
        analysis.get("vlm_reason"),
    )

    if not is_relevant:
        unwanted_categories = res.get("unwanted_categories") or []
        vlm_reason = (res.get("reason") or "").strip()

        submission.status = Submission.REFUSED
        submission.flow_stage = Submission.FLOW_REFUSED

        if recapture:
            submission.message = (
                "Submission rejected: the video appears to capture a screen, printed photo, "
                "or another reproduced image. Please record the requested real-world subject directly."
            )
        else:
            submission.message = (
                f"Submission rejected due to irrelevance (video): {vlm_reason}"
                if vlm_reason
                else "Submission rejected due to irrelevance (video)"
            )

        submission.save(update_fields=["status", "flow_stage", "message"])

        analysis["task_total_s"] = float(time.perf_counter() - t0)

        logger.info(
            "[stage_relevance_video] REFUSED submission_id=%s relevant=%s recapture=%s "
            "unwanted_categories=%s reason=%r message=%r task_total_s=%.3f",
            submission_id,
            is_relevant,
            recapture,
            unwanted_categories,
            vlm_reason,
            submission.message,
            analysis["task_total_s"],
        )

        # cleanup frames
        try:
            for f in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, f))
            os.rmdir(frames_dir)
        except Exception:
            pass

        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "analysis": analysis}

    # 2) Persist caption into Photo.caption
    try:
        video: Video = submission.video
        caption = (res.get("caption") or "").strip()
        if caption and not (video.caption or "").strip():
            video.caption = caption
            video.save(update_fields=["caption"])
    except Exception:
        pass

    # 3) Compute video embedding from 8 frames
    t_emb = time.perf_counter()
    video_emb = clip_video_embedding_from_frames_dir(frames_dir)
    analysis["embedding_extract_s"] = float(time.perf_counter() - t_emb)

    # 4) Upsert into Qdrant
    video: Video = submission.video
    from qdrant_client.models import PointStruct
    # ✅ cleanup frames ONLY when relevant ✅
    # cleanup frames
    try:
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        os.rmdir(frames_dir)
    except Exception:
        pass


    point = PointStruct(
        id=int(video.id),
        vector=video_emb,
        payload={
            "video_id": video.id,
            "submission_id": submission.id,
            "event_id": submission.event.id,
            "kind": "video",
            "frames_dir": frames_dir,
            "metadata": video.metadata,
        }
    )

    tq = time.perf_counter()
    qc = QdrantClient(
        url=getattr(settings, "QDRANT_URL", None),
        api_key=getattr(settings, "QDRANT_API_KEY", None),
    )
    qc.upsert(collection_name=settings.QDRANT_COLLECTION_VIDEO, points=[point])
    analysis["qdrant_upsert_s"] = float(time.perf_counter() - tq)

    analysis["task_total_s"] = float(time.perf_counter() - t0)

    return {"stop": False, "submission_id": submission_id, "user_id": user_id, "analysis": analysis}
from celery import shared_task
from typing import Any, Dict, Optional, Tuple
import time
import logging
from datetime import datetime, timezone
from math import radians, sin, cos, atan2, sqrt

from qdrant_client.models import Filter, FieldCondition, MatchValue

from mobicrowd.models.submisson import Submission
from mobicrowd.qdrant_service import client  # your existing global qdrant client


# ==========================================================
# Qdrant scan fallback (your version, kept as-is)
# ==========================================================
def scan_video_embedding(video_id: int):
    """
    Fallback method to retrieve the embedding vector for a video using a brute-force scan,
    when indexed filtering fails. Works reliably as long as video_id is unique.
    """
    try:
        scroll_result, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_VIDEO,
            with_vectors=True,
            with_payload=True,
            limit=1000
        )

        for point in scroll_result:
            if point.payload.get("video_id") == video_id:
                vec = getattr(point, "vector", None)
                if isinstance(vec, dict):
                    vec = vec.get("default") or next(iter(vec.values()), None)
                return vec

        logging.warning(f"⚠️ Video ID {video_id} not found in Qdrant collection.")
        return None

    except Exception as e:
        logging.error(f"❌ Error during scan for video {video_id}: {e}")
        return None


# ==========================================================
# Robust geo/time helpers (your logic, tuned for Phase-A style)
# ==========================================================
DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"  # backward compatibility


def haversine(coord1, coord2) -> float:
    """Return distance in kilometers between two (lat, lon) pairs."""
    R = 6371.0
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _parse_dt_any(x) -> Optional[datetime]:
    """
    Accepts:
      - ISO 8601: "2025-11-01T09:31:54.885Z"
      - legacy "YYYY:MM:DD HH:MM:SS"
      - unix seconds (int/float)
    Returns timezone-aware UTC datetime or None.
    """
    if not x:
        return None

    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc)

    if isinstance(x, str):
        s = x.strip()
        # ISO-8601
        try:
            if s.endswith("Z"):
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            dt = datetime.fromisoformat(s)
            # normalize if naive
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

        # legacy format
        try:
            return datetime.strptime(s, DATETIME_FORMAT).replace(tzinfo=timezone.utc)
        except Exception:
            return None

    return None


def _extract_lat_lon_dt(meta: dict) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
    """
    Expected nested shape:
      meta["capture"]["gps"]["latitude"], meta["capture"]["gps"]["longitude"]
      meta["capture"]["timestamp"] (ISO8601)

    Fallbacks:
      - meta["lat"]/meta["lon"] or meta["latitude"]/meta["longitude"]
      - meta["datetime"] or meta["timestamp"] at capture or root
    """
    lat = lon = None
    dt = None

    if not isinstance(meta, dict):
        return None, None, None

    cap = meta.get("capture") or {}
    gps = cap.get("gps") or {}

    # latitude
    for k in ("latitude", "lat"):
        if gps.get(k) is not None:
            try:
                lat = float(gps[k])
            except Exception:
                pass
            break

    # longitude
    for k in ("longitude", "lon"):
        if gps.get(k) is not None:
            try:
                lon = float(gps[k])
            except Exception:
                pass
            break

    # root fallback
    if lat is None:
        v = meta.get("latitude") or meta.get("lat")
        if v is not None:
            try:
                lat = float(v)
            except Exception:
                pass

    if lon is None:
        v = meta.get("longitude") or meta.get("lon")
        if v is not None:
            try:
                lon = float(v)
            except Exception:
                pass

    # timestamp
    ts = cap.get("timestamp") or meta.get("timestamp") or meta.get("datetime")
    dt = _parse_dt_any(ts)

    return lat, lon, dt


# ==========================================================
# ✅ Phase-A redundancy task (drop-in replacement)
# ==========================================================
@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=300,
    time_limit=360,
)
def stage_redundancy_video(
    self,
    prev: Dict[str, Any],
    *,
    threshold: float = 0.87,
    location_threshold: float = 0.05,  # km (50m)
    time_threshold: int = 86400,       # seconds (24h)
) -> Dict[str, Any]:
    """
    VIDEO redundancy criteria (Phase A):
      1) Similarity score >= threshold and different video_id
      2) Within location_threshold (km)
      3) Time difference <= time_threshold (seconds)

    If redundant:
      - REFUSE submission
      - delete current video's point from Qdrant (optional but recommended)
      - return {"stop": True, ...}

    If NOT redundant:
      - DO NOT approve (keep submission pending for Phase B upload/finalize)
      - return prev with is_redundant False
    """

    if prev.get("stop"):
        return prev

    submission_id = int(prev["submission_id"])
    user_id = int(prev["user_id"])

    submission = Submission.objects.select_related("video", "event", "worker", "worker__user").get(id=submission_id)
    if submission.worker.user_id != user_id:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "owner_mismatch"}
    if submission.status == Submission.REFUSED:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "already_refused"}
    if submission.flow_stage != Submission.FLOW_VALIDATING:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "invalid_flow_state"}

    video = submission.video
    if not video:
        raise RuntimeError("Submission has no video object linked yet.")

    metadata = video.metadata or {}

    prev.setdefault("analysis", {})
    analysis = prev["analysis"]
    t0 = time.perf_counter()

    # -----------------------------
    # current item gps/time
    # -----------------------------
    cur_lat, cur_lon, cur_dt = _extract_lat_lon_dt(metadata)

    # If GPS is missing, we cannot apply your redundancy definition reliably.
    # Treat as "not redundant" but record diagnostic info.
    if cur_lat is None or cur_lon is None:
        analysis["redundancy"] = {
            "redundant": False,
            "reason": "missing_gps",
            "threshold": float(threshold),
        }
        prev["is_redundant"] = False
        return prev

    if cur_dt is None:
        # best-effort fallback
        cur_dt = datetime.now(timezone.utc)

    # -----------------------------
    # fetch embedding (from qdrant)
    # -----------------------------
    embedding = scan_video_embedding(video.id)
    if not embedding:
        # retryable: stage_relevance probably failed to upsert or qdrant delayed
        raise RuntimeError("Embedding not found in Qdrant for video redundancy check")

    # -----------------------------
    # qdrant search in same event
    # -----------------------------
    t_search = time.perf_counter()
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION_VIDEO,
        query_vector=embedding,
        limit=10,
        query_filter=Filter(
            must=[
                FieldCondition(key="event_id", match=MatchValue(value=int(submission.event.id)))
            ]
        ),
    )
    analysis["qdrant_search_s"] = float(time.perf_counter() - t_search)

    # Similar by score + not same video
    similar = [
        p for p in results
        if float(p.score) >= float(threshold)
        and str((p.payload or {}).get("video_id")) != str(video.id)
    ]

    if not similar:
        analysis["redundancy"] = {
            "redundant": False,
            "hits_over_threshold": 0,
            "threshold": float(threshold),
        }
        prev["is_redundant"] = False
        return prev

    # -----------------------------
    # apply geo + time filters
    # -----------------------------
    nearby_candidates = []
    matches_out = []

    for p in similar:
        p_payload = p.payload or {}
        p_meta = p_payload.get("metadata") or {}

        plat, plon, pdt = _extract_lat_lon_dt(p_meta)

        diag = {
            "video_id": p_payload.get("video_id"),
            "submission_id": p_payload.get("submission_id"),
            "event_id": p_payload.get("event_id"),
            "similarity_score": round(float(p.score), 4),
            "distance_km": None,
            "time_delta_sec": None,
        }

        dist_km = None
        if plat is not None and plon is not None:
            dist_km = haversine((cur_lat, cur_lon), (plat, plon))
            diag["distance_km"] = round(dist_km, 6)

        td = None
        if pdt is not None:
            td = abs((cur_dt - pdt).total_seconds())
            diag["time_delta_sec"] = round(td, 3)

        matches_out.append(diag)

        if (
            dist_km is not None and dist_km <= float(location_threshold)
            and td is not None and td <= float(time_threshold)
        ):
            nearby_candidates.append((p, dist_km, td, pdt))

    # -----------------------------
    # decision
    # -----------------------------
    if nearby_candidates:
        # choose most recent spatiotemporal match
        best = max(
            nearby_candidates,
            key=lambda t: (t[3] or datetime.fromtimestamp(0, tz=timezone.utc))
        )
        best_point, best_dist, best_td, best_dt = best

        # REFUSE
        submission.status = Submission.REFUSED
        submission.flow_stage = Submission.FLOW_REFUSED
        submission.message = "Submission rejected due to redundancy (video)"
        submission.save(update_fields=["status", "flow_stage", "message"])

        # delete current video's point so it won't pollute future checks
        try:
            client.delete(
                collection_name=settings.QDRANT_COLLECTION_VIDEO,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="video_id", match=MatchValue(value=int(video.id)))
                    ]
                ),
            )
        except Exception:
            pass

        analysis["redundancy"] = {
            "redundant": True,
            "threshold": float(threshold),
            "location_threshold_km": float(location_threshold),
            "time_threshold_sec": int(time_threshold),
            "distance_to_match_km": round(float(best_dist), 6),
            "time_diff_sec": round(float(best_td), 3),
            "most_recent_match": {
                **(best_point.payload or {}),
                "similarity_score": round(float(best_point.score), 4),
            },
            "matches": matches_out,
        }
        analysis["task_total_s"] = float(time.perf_counter() - t0)

        return {
            "stop": True,
            "submission_id": submission_id,
            "user_id": user_id,
            "is_redundant": True,
            "analysis": analysis,
        }

    # not redundant => DO NOT approve here
    analysis["redundancy"] = {
        "redundant": False,
        "threshold": float(threshold),
        "location_threshold_km": float(location_threshold),
        "time_threshold_sec": int(time_threshold),
        "hits_over_threshold": len(similar),
        "matches": matches_out,
    }
    analysis["task_total_s"] = float(time.perf_counter() - t0)

    prev["is_redundant"] = False
    return prev



def _decode_base64_video(base64_video: str) -> bytes:
    # Accept "data:video/mp4;base64,...." or raw b64
    if not base64_video:
        return b""
    if "," in base64_video:
        _, b64 = base64_video.split(",", 1)
    else:
        b64 = base64_video
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 video payload") from exc


def _cosine_similarity(a, b) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.shape != bv.shape or av.size == 0:
        raise ValueError("Video embedding shape mismatch")
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an <= 0 or bn <= 0:
        raise ValueError("Cannot compare zero-norm video embeddings")
    return float(np.dot(av, bv) / (an * bn))


def _extract_uniform_video_frames(src_path: str, out_dir: str, count: int = 8) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError("Cannot decode uploaded original video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError("Uploaded video has no decodable frames")

    indices = np.linspace(0, max(total - 1, 0), count, dtype=int).tolist()
    try:
        for out_idx, frame_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Cannot read validation frame {frame_idx}")
            target = root / f"{out_idx}.jpeg"
            if not cv2.imwrite(str(target), frame):
                raise ValueError(f"Cannot write validation frame {out_idx}")
    finally:
        cap.release()


def _delete_current_video_embedding(video_id: int) -> None:
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_VIDEO,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=int(video_id)))]
            ),
        )
    except Exception:
        logger.exception("Failed to delete video embedding video_id=%s", video_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def process_original_video_and_finalize(
    self,
    *,
    submission_id: int,
    user_id: int,
    base64_video: str,
    file_name: str,
    flow_nonce: str,
) -> Dict[str, Any]:
    """Secure Phase-B video finalizer with state, nonce, content-binding and quota checks."""
    submission = Submission.objects.select_related(
        "event", "worker", "worker__user", "video"
    ).get(id=submission_id)

    if submission.worker.user_id != int(user_id):
        return {"ok": False, "reason": "owner_mismatch"}
    if str(submission.flow_nonce) != str(flow_nonce):
        return {"ok": False, "reason": "stale_flow_nonce"}
    if submission.status != Submission.PENDING:
        return {"ok": False, "reason": f"invalid_status:{submission.status}"}
    if submission.flow_stage != Submission.FLOW_FINALIZING:
        return {"ok": False, "reason": f"invalid_flow_state:{submission.flow_stage}"}
    if not submission.video_id:
        return {"ok": False, "reason": "missing_video"}
    if not base64_video or not file_name:
        raise ValueError("base64_video and file_name are required")

    video_obj = submission.video
    tmp_dir = tempfile.mkdtemp(prefix=f"vidproc_{uuid.uuid4().hex[:8]}_")
    try:
        raw = _decode_base64_video(base64_video)
        if not raw:
            raise ValueError("Failed to decode base64 video (empty bytes)")
        src_path = os.path.join(tmp_dir, f"input_{uuid.uuid4().hex}.mp4")
        with open(src_path, "wb") as f:
            f.write(raw)

        # Bind the original upload to the exact Phase-A validation identity.
        validated_embedding = scan_video_embedding(video_obj.id)
        if not validated_embedding:
            raise RuntimeError("Validated video embedding is missing from Qdrant")
        binding_frames = os.path.join(tmp_dir, "binding_frames")
        _extract_uniform_video_frames(src_path, binding_frames, count=8)
        original_embedding = clip_video_embedding_from_frames_dir(binding_frames)
        binding_similarity = _cosine_similarity(validated_embedding, original_embedding)
        binding_threshold = configured_binding_threshold(
            "SUBMISSION_VIDEO_ORIGINAL_BINDING_MIN_COSINE"
        )

        if binding_similarity < binding_threshold:
            with transaction.atomic():
                locked = Submission.objects.select_for_update().get(pk=submission_id)
                if (
                    locked.status == Submission.PENDING
                    and locked.flow_stage == Submission.FLOW_FINALIZING
                    and str(locked.flow_nonce) == str(flow_nonce)
                ):
                    mark_refused(
                        submission=locked,
                        message="Submission rejected: uploaded original does not match the validated video.",
                    )
            _delete_current_video_embedding(video_obj.id)
            signal_user(
                user_id=user_id,
                signal="submission.finalized",
                payload={
                    "submission_id": submission_id,
                    "event_id": submission.event_id,
                    "decision": "REJECT",
                    "reason": "original_mismatch",
                },
            )
            return {
                "ok": False,
                "reason": "original_mismatch",
                "binding_similarity": binding_similarity,
            }

        with transaction.atomic():
            locked = (
                Submission.objects.select_for_update()
                .select_related("event", "worker", "worker__user", "video")
                .get(pk=submission_id)
            )
            locked.event = Event.objects.select_for_update().get(pk=locked.event_id)
            if locked.status != Submission.PENDING or locked.flow_stage != Submission.FLOW_FINALIZING:
                return {"ok": False, "reason": "state_changed_before_upload"}
            if str(locked.flow_nonce) != str(flow_nonce):
                return {"ok": False, "reason": "stale_flow_nonce"}
            capacity_ok, capacity_reason = approved_capacity_available(locked)
            if not capacity_ok:
                mark_refused(submission=locked, message="Submission rejected: event submission capacity reached.")
                _delete_current_video_embedding(video_obj.id)
                return {"ok": False, "reason": capacity_reason}

        # Face blur.
        out_path = os.path.join(tmp_dir, f"blurred_{uuid.uuid4().hex}.mp4")
        stats = blur_video_faces_sparse(
            in_path=src_path,
            out_path=out_path,
            stride=1,
            margin=0.25,
            conf=0.25,
            imgsz=640,
            blur_ksize=31,
            writer_fourcc="mp4v",
        )

        full_blurred_path = os.path.join(tmp_dir, f"blurred_full_{uuid.uuid4().hex}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-movflags", "+faststart", full_blurred_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        preview_path = os.path.join(tmp_dir, f"preview_{uuid.uuid4().hex}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_path,
                "-vf", "scale=854:480", "-b:v", "800k", "-c:v", "libx264",
                "-preset", "fast", "-movflags", "+faststart", preview_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        full_bytes = Path(full_blurred_path).read_bytes()
        preview_bytes = Path(preview_path).read_bytes()

        s3_vid = CustomS3Boto3Storage(
            folder_name="submissions",
            dynamic_folder_name=f"{submission.event.title}/{submission.worker.user.fullName}/{submission.id}",
        )
        s3_thumb = CustomS3Boto3Storage(
            folder_name="thumbs",
            dynamic_folder_name=s3_vid.dynamic_folder_name,
        )
        video_key = s3_vid.save(file_name, ContentFile(full_bytes, name=file_name))
        preview_name = f"preview_{Path(file_name).stem}.mp4"
        preview_key = s3_thumb.save(preview_name, ContentFile(preview_bytes, name=preview_name))

        video_obj.video = video_key
        video_obj.thumbnail = preview_key
        video_obj.size = len(full_bytes) / (1024 * 1024.0)
        video_obj.save(update_fields=["video", "thumbnail", "size"])

        total_size_mb = (len(full_bytes) + len(preview_bytes)) / (1024 * 1024.0)
        user_log, _ = UserUploadLog.objects.get_or_create(
            submission=submission,
            defaults={"user": submission.worker.user, "size_mb": 0},
        )
        user_log.user = submission.worker.user
        user_log.size_mb = (user_log.size_mb or 0) + total_size_mb
        user_log.save(update_fields=["user", "size_mb"])

        with transaction.atomic():
            locked = (
                Submission.objects.select_for_update()
                .select_related("event", "worker", "worker__user")
                .get(pk=submission_id)
            )
            locked.event = Event.objects.select_for_update().get(pk=locked.event_id)
            if locked.status != Submission.PENDING:
                return {"ok": False, "reason": f"status_changed:{locked.status}"}
            if locked.flow_stage != Submission.FLOW_FINALIZING:
                return {"ok": False, "reason": f"flow_changed:{locked.flow_stage}"}
            if str(locked.flow_nonce) != str(flow_nonce):
                return {"ok": False, "reason": "stale_flow_nonce"}
            capacity_ok, capacity_reason = approved_capacity_available(locked)
            if not capacity_ok:
                mark_refused(submission=locked, message="Submission rejected: event submission capacity reached.")
                _delete_current_video_embedding(video_obj.id)
                return {"ok": False, "reason": capacity_reason}

            locked.status = Submission.APPROVED
            locked.flow_stage = Submission.FLOW_FINALIZED
            locked.flow_finalized_at = dj_timezone.now()
            locked.message = "Submission accepted (video)"
            locked.save(
                update_fields=["status", "flow_stage", "flow_finalized_at", "message"]
            )

        corr_id = signal_user(
            user_id=user_id,
            signal="submission.finalized",
            payload={
                "submission_id": submission_id,
                "event_id": submission.event_id,
                "event_title": submission.event.title,
                "decision": "ACCEPT",
                "worker_id": submission.worker.user_id,
            },
        )
        logger.info(
            "[WS] video finalized corr=%s submission_id=%s binding_similarity=%.5f",
            corr_id,
            submission_id,
            binding_similarity,
        )
        return {
            "ok": True,
            "video_s3": video_key,
            "preview_s3": preview_key,
            "total_size_mb": round(total_size_mb, 3),
            "frames": stats.get("frames"),
            "detect_calls": stats.get("detect_calls"),
            "binding_similarity": binding_similarity,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

