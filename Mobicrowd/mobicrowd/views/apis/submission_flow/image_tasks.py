from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from celery import shared_task, chain
from django.core import signing
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from qdrant_client.models import Filter, FieldCondition, MatchValue
from mobicrowd.models.submisson import Submission, Photo, Event
from mobicrowd.models.FileStorage import CustomS3Boto3Storage
from mobicrowd.tasks import clip_processor, device, clip_model
from mobicrowd.notify import notify_user, signal_user
from mobicrowd.qdrant_service import client
from mobicrowd.views.apis.submission_flow.security import (
    approved_capacity_available,
    configured_binding_threshold,
    mark_refused,
)
from django.core.files.base import ContentFile
from PIL import Image, ImageOps
from io import BytesIO

from mobicrowd.views.apis.submission_flow.qwen3_caption_relevance import openrouter_qwen3_vl_caption_and_relevance
from mobicrowd.views.apis.submission_views import save_to_cache
import time
import random
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_RELEVANCE_CTX: Optional[Dict[str, Any]] = None

import logging
from celery.utils.log import get_task_logger

logger = get_task_logger("mobicrowd.submission_flow")

def _log_start(task, name: str, **fields):
    logger.info("[%s] start task_id=%s %s", name, task.request.id, fields)

def _log_end(name: str, result):
    logger.info("[%s] end result=%s", name, result)

def scan_photo_embedding(photo_id: int):
    """
    Fallback method to retrieve the embedding vector for a photo using a brute-force scan,
    when indexed filtering fails. Works reliably as long as photo_id is unique.
    """
    try:
        scroll_result, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            with_vectors=True,
            with_payload=True,
            limit=1000  # you can increase this if needed
        )

        for point in scroll_result:
            if point.payload.get("photo_id") == photo_id:
                return point.vector

        logging.warning(f"⚠️ Photo ID {photo_id} not found in Qdrant collection.")
        return None

    except Exception as e:
        logging.error(f"❌ Error during scan for photo {photo_id}: {e}")
        return None

BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
CACHE_DIR = BASE_DIR / "tmp" / "blurred-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
YOLO_MODEL_PATH = "mobicrowd/models/UNIQUE/yolov8n-face.pt"  # Update this path

def load_yolo_model(path=YOLO_MODEL_PATH):
    return YOLO(path)

def encode_cv2_image_to_base64(cv2_img):
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_img}"
def blur_faces(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face, (171, 131), 60)
        image[y1:y2, x1:x2] = blurred_face
    return image
def blur_faces_pipeline_from_base64(base64_image_str, filename_prefix="blurred"):
    import numpy as np
    from datetime import datetime

    # Decode base64 image string
    base64_data = base64_image_str.split(',')[1] if ',' in base64_image_str else base64_image_str
    image_data = base64.b64decode(base64_data)
    nparr = np.frombuffer(image_data, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Failed to decode image from base64 string.")

    model = load_yolo_model()
    results = model.predict(image_bgr)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    final_image = blur_faces(image_bgr, boxes) if len(boxes) > 0 else image_bgr
    base64_img = encode_cv2_image_to_base64(final_image)

    # Generate a fake path string (optional)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{filename_prefix}_{timestamp}.jpg"
    output_path = str(CACHE_DIR / output_filename)

    return {
        "blurred": len(boxes) > 0,
        "base64": base64_img,
        "image_bytes": base64.b64decode(base64_img.split(",")[1]),
        "saved_path": output_path  # optional if you later want to log
    }

# -----------------------------
# Token + cache handshake
# -----------------------------
ORIG_REQ_TTL_SECONDS = 60 * 60  # 1 hour (adjust)
SIGN_SALT = "submission-original-upload"


def _cache_key(submission_id: int) -> str:
    return f"submission:orig:req:{submission_id}"


def make_upload_token(*, submission_id: int, user_id: int, nonce: str) -> str:
    signer = signing.TimestampSigner(salt=SIGN_SALT)
    return signer.sign(f"{submission_id}:{user_id}:{nonce}")


def verify_upload_token(*, token: str, max_age_seconds: int = ORIG_REQ_TTL_SECONDS) -> Tuple[int, int, str]:
    signer = signing.TimestampSigner(salt=SIGN_SALT)
    raw = signer.unsign(token, max_age=max_age_seconds)
    submission_id_s, user_id_s, nonce = raw.split(":")
    return int(submission_id_s), int(user_id_s), nonce


def extract_vision_embeddings(image_tensor, model_dir: str, device: str):
    """
    Runs OpenVINO vision-embeddings model:
      model_dir/openvino_vision_embeddings_model.xml

    Returns numpy array (e.g. (1, 729, 1024)) or None.
    """
    try:
        from openvino.runtime import Core  # local import to avoid import-time failures

        vision_model_path = os.path.join(model_dir, "openvino_vision_embeddings_model.xml")
        if not os.path.exists(vision_model_path):
            logger.error("Vision model not found: %s", vision_model_path)
            return None

        core = Core()
        ov_model = core.read_model(vision_model_path)
        compiled = core.compile_model(ov_model, device)
        output_layer = compiled.output(0)

        # Convert to numpy
        if not isinstance(image_tensor, np.ndarray):
            if hasattr(image_tensor, "numpy"):
                image_tensor = image_tensor.numpy()
            elif hasattr(image_tensor, "cpu"):
                image_tensor = image_tensor.cpu().numpy()
            else:
                image_tensor = np.array(image_tensor)

        out = compiled([image_tensor])[output_layer]
        if not isinstance(out, np.ndarray):
            out = np.array(out)

        return out
    except Exception:
        logger.exception("Error extracting vision embeddings")
        return None


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / norm if norm > 0 else embeddings
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def weighted_pooling(embeddings_2d: np.ndarray) -> np.ndarray:
    """
    embeddings_2d: (seq_len, dim)
    returns: (dim,)
    """
    weights = np.linalg.norm(embeddings_2d, axis=1, keepdims=True)
    weights = weights / (np.sum(weights) + 1e-8)
    return np.sum(embeddings_2d * weights, axis=0)


def prepare_embedding_for_db(raw: np.ndarray) -> np.ndarray:
    """
    Example:
      (1, 729, 1024) -> (1024,)
    """
    if raw is None:
        raise ValueError("raw embeddings is None")

    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]  # (seq_len, dim)

    if raw.ndim == 2:
        pooled = weighted_pooling(raw)
    elif raw.ndim == 1:
        pooled = raw
    else:
        raise ValueError(f"Unexpected embedding shape: {raw.shape}")

    pooled = normalize_embeddings(pooled).astype(np.float32, copy=False)
    return pooled

# -----------------------------
# Phase A: relevance
# -----------------------------
# def extract_caption_from_vlm_completion(completion: Optional[str]) -> str:
#     """
#     completion can be:
#       - strict JSON: {"caption_detailed": "...", "decision": "...", "reason": "..."}
#       - plain text (fallback)
#     Returns a safe caption string.
#     """
#     if not completion:
#         return ""
#
#     text = str(completion).strip()
#     if not text:
#         return ""
#
#     # Try JSON first
#     try:
#         obj = json.loads(text)
#         if isinstance(obj, dict):
#             cap = obj.get("caption_detailed") or obj.get("caption") or ""
#             return str(cap).strip()
#     except Exception:
#         pass
#
#     # Fallback: store raw completion
#     return text

# @shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
# def stage_relevance(self, *, submission_id: int, user_id: int, reconstructed_path: str, description: str) -> Dict[str, Any]:
#     t_task0 = time.perf_counter()
#
#     submission = Submission.objects.select_related("photo", "event").get(pk=submission_id)
#
#     # Idempotency
#     if submission.status in (Submission.APPROVED, Submission.REFUSED):
#         return {
#             "stop": True,
#             "submission_id": submission_id,
#             "user_id": user_id,
#             "reason": "already_final",
#         }
#
#     # Hard fail if missing reconstructed image
#     if not reconstructed_path or not os.path.exists(reconstructed_path):
#         submission.status = Submission.REFUSED
#         submission.message = "Submission rejected: missing reconstructed image"
#         submission.save(update_fields=["status", "flow_stage", "message"])
#
#         return {
#             "stop": True,
#             "submission_id": submission_id,
#             "user_id": user_id,
#             "analysis": {
#                 "is_relevant": False,
#                 "reason": "missing_reconstructed",
#                 "task_total_s": float(time.perf_counter() - t_task0),
#             },
#         }
#
#     # ---- VLM relevance (and keep pixel_values for embeddings) ----
#     is_relevant, generate_s, completion, timings, pixel_values = openvino_vlm_relevance(
#         image_path=reconstructed_path,
#         description=description or "",
#         model_dir="mobicrowd/models/FP16",
#         device=getattr(settings, "OV_VLM_DEVICE", "CPU"),
#         max_new_tokens=70,
#         return_pixel_values=True,
#     )
#
#     analysis: Dict[str, Any] = {
#         "is_relevant": bool(is_relevant),
#         # VLM timing breakdown
#         "vlm_generate_s": float(generate_s or 0.0),                 # time inside model.generate
#         "vlm_preprocess_s": float(timings.get("preprocess_s") or 0.0),
#         "vlm_decode_s": float(timings.get("decode_s") or 0.0),
#         "vlm_total_s": float(timings.get("total_s") or 0.0),        # preprocess+generate+decode
#         "vlm_completion": completion [:800],
#     }
#     # Cleanup reconstructed image (safe after pixel_values exists)
#
#
#     # ---- reject if not relevant ----
#     if not is_relevant:
#         submission.status = Submission.REFUSED
#         submission.message = "Submission rejected due to irrelevance"
#         submission.save(update_fields=["status", "message"])
#
#         analysis["task_total_s"] = float(time.perf_counter() - t_task0)
#         try:
#             logger.info(reconstructed_path)
#             os.remove(reconstructed_path)
#         except Exception:
#             pass
#         return {
#             "stop": True,
#             "submission_id": submission_id,
#             "user_id": user_id,
#         }
#
#     # ---- embeddings extraction + qdrant upsert ----
#     photo: Photo = submission.photo
#
#
#     t_emb0 = time.perf_counter()
#     # ✅ Persist caption from completion for Stage B
#     try:
#         photo: Photo = submission.photo
#         caption_from_vlm = extract_caption_from_vlm_completion(completion)
#
#         # Avoid overwriting if already set (optional)
#         if caption_from_vlm and not (photo.caption or "").strip():
#             photo.caption = caption_from_vlm
#             photo.save(update_fields=["caption"])
#     except Exception:
#         # Caption persistence should never break stage A
#         pass
#     raw_emb = extract_vision_embeddings(
#         pixel_values,
#         model_dir="mobicrowd/models/INT4",  # contains openvino_vision_embeddings_model.xml
#         device=getattr(settings, "OV_VLM_DEVICE", "CPU"),
#     )
#     if raw_emb is None:
#         # Treat as internal failure (lets autoretry work in async mode; in sync mode your API should return 500)
#         raise RuntimeError("Vision embeddings extraction failed")
#
#     pooled = prepare_embedding_for_db(raw_emb)  # (dim,)
#     analysis["embedding_dim"] = int(pooled.shape[0])
#     analysis["embedding_extract_s"] = float(time.perf_counter() - t_emb0)
#
#     # Ensure payload is JSON-serializable (avoid Qdrant serialization errors)
#     payload_metadata = photo.original_metadata or {}
#     try:
#         payload_metadata = json.loads(json.dumps(payload_metadata, default=str))
#     except Exception:
#         payload_metadata = {}
#
#     point_id = int(photo.id) if str(photo.id).isdigit() else str(photo.id)
#     from qdrant_client.models import PointStruct
#     point = PointStruct(
#         id=point_id,
#         vector=pooled.tolist(),
#         payload={
#             "photo_id": int(photo.id) if str(photo.id).isdigit() else str(photo.id),
#             "submission_id": int(submission.id),
#             "event_id": int(submission.event.id),
#             "metadata": payload_metadata,
#         },
#     )
#
#     t_q0 = time.perf_counter()
#     client.upsert(collection_name=settings.QDRANT_COLLECTION, points=[point])
#     analysis["qdrant_upsert_s"] = float(time.perf_counter() - t_q0)
#
#     # Done
#     analysis["task_total_s"] = float(time.perf_counter() - t_task0)
#
#     logger.info(
#         "[stage_relevance] done submission_id=%s task_total_s=%.3f embedding_dim=%s",
#         submission_id,
#         analysis["task_total_s"],
#         analysis.get("embedding_dim"),
#     )
#     try:
#         logger.info(reconstructed_path)
#         os.remove(reconstructed_path)
#     except Exception:
#         pass
#
#     return {
#         "stop": False,
#         "submission_id": submission_id,
#         "user_id": user_id,
#         "analysis": analysis,
#     }




@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def stage_relevance(
    self,
    *,
    submission_id: int,
    user_id: int,
    reconstructed_path: str,
    description: str
) -> Dict[str, Any]:
    t_task0 = time.perf_counter()

    submission = Submission.objects.select_related("photo", "event", "worker", "worker__user").get(pk=submission_id)

    if submission.worker.user_id != int(user_id):
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "owner_mismatch"}
    if submission.flow_stage != Submission.FLOW_VALIDATING:
        return {
            "stop": True,
            "submission_id": submission_id,
            "user_id": user_id,
            "reason": "invalid_flow_state",
            "flow_stage": submission.flow_stage,
        }

    # -----------------------------
    # Idempotency
    # -----------------------------
    if submission.status in (Submission.APPROVED, Submission.REFUSED):
        return {
            "stop": True,
            "submission_id": submission_id,
            "user_id": user_id,
            "reason": "already_final",
        }

    # -----------------------------
    # Hard fail if missing reconstructed image
    # -----------------------------
    if not reconstructed_path or not os.path.exists(reconstructed_path):
        submission.status = Submission.REFUSED
        submission.flow_stage = Submission.FLOW_REFUSED
        submission.message = "Submission rejected: missing reconstructed image"
        submission.save(update_fields=["status", "flow_stage", "message"])

        return {
            "stop": True,
            "submission_id": submission_id,
            "user_id": user_id,
            "analysis": {
                "is_relevant": False,
                "reason": "missing_reconstructed",
                "task_total_s": float(time.perf_counter() - t_task0),
            },
        }

    # ==========================================================
    # ✅ Stage A: OpenRouter Qwen3-VL relevance + caption
    # ==========================================================
    t_vlm0 = time.perf_counter()

    keys_file = settings.OPENROUTER_KEYS_FILE
    model_id = settings.OPENROUTER_IMAGE_RELEVANCE_MODEL
    fallback_models = list(
        getattr(settings, "OPENROUTER_IMAGE_RELEVANCE_FALLBACK_MODELS", [])
    )

    if not keys_file or not os.path.exists(keys_file):
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {keys_file}")
    if not model_id:
        raise RuntimeError("OPENROUTER_IMAGE_RELEVANCE_MODEL is not configured.")

    res = openrouter_qwen3_vl_caption_and_relevance(
        image_path=reconstructed_path,
        description=(description or "").strip(),
        keys_file=keys_file,
        logger=logger,
        model_id=model_id,
        fallback_models=fallback_models,
        max_attempts=6,
        max_tokens=320,
        jpeg_quality=75,
    )

    if not res.get("ok"):
        raise RuntimeError(
            "OpenRouter image relevance failed after configured model fallbacks: "
            f"{res.get('reason')} / err={res.get('error')}"
        )

    # Canonical recapture flag propagated by the VLM helper.
    # A confirmed recapture is always treated as not relevant.
    recapture = bool(res.get("recapture", False))
    is_relevant = (res.get("relevant") == "YES") and not recapture

    analysis: Dict[str, Any] = {
        "is_relevant": bool(is_relevant),
        "recapture": recapture,
        "description": description,
        "vlm_total_s": float(time.perf_counter() - t_vlm0),
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
        "[stage_relevance] VLM_DECISION submission_id=%s relevant=%s recapture=%s "
        "unwanted_content=%s unwanted_categories=%s reason=%r",
        submission_id,
        is_relevant,
        recapture,
        analysis.get("vlm_unwanted_content"),
        analysis.get("vlm_unwanted_categories"),
        analysis.get("vlm_reason"),
    )

    # -----------------------------
    # Reject if not relevant
    # -----------------------------
    if not is_relevant:
        unwanted_categories = res.get("unwanted_categories") or []
        vlm_reason = (res.get("reason") or "").strip()

        submission.status = Submission.REFUSED
        submission.flow_stage = Submission.FLOW_REFUSED

        if recapture:
            submission.message = (
                "Submission rejected: the image appears to be a photo of a screen, "
                "printed photo, or another reproduced image. Please capture the "
                "requested real-world subject directly."
            )
        elif vlm_reason:
            submission.message = f"Submission rejected due to irrelevance: {vlm_reason}"
        else:
            submission.message = "Submission rejected due to irrelevance"

        submission.save(update_fields=["status", "flow_stage", "message"])

        analysis["task_total_s"] = float(time.perf_counter() - t_task0)

        logger.info(
            "[stage_relevance] REFUSED submission_id=%s description=%r relevant=%s "
            "recapture=%s unwanted_categories=%s reason=%r message=%r task_total_s=%.3f",
            submission_id,
            analysis.get("description"),
            is_relevant,
            recapture,
            unwanted_categories,
            vlm_reason,
            submission.message,
            analysis["task_total_s"],
        )

        # Cleanup reconstructed image
        try:
            os.remove(reconstructed_path)
        except Exception:
            pass

        return {
            "stop": True,
            "submission_id": submission_id,
            "user_id": user_id,
            "analysis": analysis,
        }

    # ==========================================================
    # ✅ Stage B: Persist caption into Photo.caption (from OpenRouter)
    # ==========================================================
    try:
        photo: Photo = submission.photo
        caption_from_vlm = (res.get("caption") or "").strip()
        if caption_from_vlm and not (photo.caption or "").strip():
            photo.caption = caption_from_vlm
            photo.save(update_fields=["caption"])
    except Exception:
        # Caption persistence must never break the task
        pass

    # ==========================================================
    # ✅ Embeddings extraction (still OpenVINO pixel_values based)
    # ==========================================================
    # Because your embeddings code expects pixel_values, we run OpenVINO only to get pixel_values.
    # This is separate from relevance now.
    t_emb0 = time.perf_counter()

    img = Image.open(reconstructed_path).convert("RGB")

    inputs = clip_processor(
        text="prompts",
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=clip_processor.tokenizer.model_max_length,
    ).to(device)
    image_emb = clip_model(**inputs).image_embeds[0].tolist()

    analysis["embedding_extract_s"] = float(time.perf_counter() - t_emb0)

    # ==========================================================
    # ✅ Qdrant upsert
    # ==========================================================
    photo: Photo = submission.photo

    payload_metadata = photo.original_metadata or {}
    try:
        payload_metadata = json.loads(json.dumps(payload_metadata, default=str))
    except Exception:
        payload_metadata = {}

    point_id = int(photo.id) if str(photo.id).isdigit() else str(photo.id)

    from qdrant_client.models import PointStruct
    point = PointStruct(
        id=point_id,  # must be unique
        vector=image_emb,
        payload={
            "photo_id": photo.id,
            "submission_id": submission.id,
            "event_id": submission.event.id,
            "metadata": photo.original_metadata
        }
    )

    t_q0 = time.perf_counter()

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=[point])
    analysis["qdrant_upsert_s"] = float(time.perf_counter() - t_q0)

    # Done
    analysis["task_total_s"] = float(time.perf_counter() - t_task0)

    logger.info(
        "[stage_relevance] done submission_id=%s relevant=%s embedding_extract_s=%.3f task_total_s=%.3f tokens=%s cost=%s",
        submission_id,
        True,
        analysis["embedding_extract_s"],
        analysis["task_total_s"],
        (analysis.get("vlm_usage") or {}).get("total_tokens"),
        analysis.get("vlm_cost"),
    )

    # Cleanup reconstructed image
    try:
        os.remove(reconstructed_path)
    except Exception:
        pass

    return {
        "stop": False,
        "submission_id": submission_id,
        "user_id": user_id,
        "analysis": analysis,
    }

# -----------------------------
# Phase A: redundancy (NO premature approval)
# -----------------------------
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3,  soft_time_limit=300,   # 5 minutes
    time_limit=360)
def stage_redundancy(self, prev: Dict[str, Any], *, threshold: float = 0.87, location_threshold: float = 0.05, time_threshold: int = 86400) -> Dict[str, Any]:
    if prev.get("stop"):
        return prev

    submission_id = int(prev["submission_id"])
    user_id = int(prev["user_id"])

    submission = Submission.objects.select_related("photo", "event", "worker", "worker__user").get(id=submission_id)
    if submission.worker.user_id != user_id:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "owner_mismatch"}
    if submission.status == Submission.REFUSED:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "already_refused"}
    if submission.flow_stage != Submission.FLOW_VALIDATING:
        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "reason": "invalid_flow_state"}
    photo = submission.photo
    metadata = photo.original_metadata or {}

    embedding = scan_photo_embedding(photo.id)
    if not embedding:
        # If we cannot validate redundancy, treat as processing failure (retryable)
        raise RuntimeError("Embedding not found in Qdrant for redundancy check")

    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=embedding,
        limit=10,
        query_filter=Filter(
            must=[FieldCondition(key="event_id", match=MatchValue(value=int(submission.event.id)))]
        ),
    )
    # logger.info(results)

    filtered = [
        p for p in results
        if float(p.score) >= float(threshold) and str(p.payload.get("photo_id")) != str(photo.id)
    ]

    # If no similar hits, not redundant
    if not filtered:
        prev["is_redundant"] = False
        return prev

    # Centroid + time logic (same as your current function)
    from datetime import datetime
    DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"

    xyz_accepted = [
        [float(p.payload["metadata"].get("x", 0)), float(p.payload["metadata"].get("y", 0)), float(p.payload["metadata"].get("z", 0))]
        for p in filtered if p.payload.get("metadata")
    ]

    most_recent_point = max(
        filtered,
        key=lambda p: datetime.strptime(p.payload["metadata"].get("datetime", "1900:01:01 00:00:00"), DATETIME_FORMAT),
    ) if filtered else None

    is_redundant = False
    if xyz_accepted and most_recent_point:
        avg_xyz = np.mean(xyz_accepted, axis=0)
        current_xyz = np.array([float(metadata.get("x", 0)), float(metadata.get("y", 0)), float(metadata.get("z", 0))])
        distance = float(np.linalg.norm(current_xyz - avg_xyz))

        if distance <= float(location_threshold):
            current_time = datetime.strptime(metadata.get("datetime", "1900:01:01 00:00:00"), DATETIME_FORMAT)
            most_recent_time = datetime.strptime(most_recent_point.payload["metadata"]["datetime"], DATETIME_FORMAT)
            time_diff = float((current_time - most_recent_time).total_seconds())

            if time_diff <= float(time_threshold):
                is_redundant = True

    if is_redundant:
        # Reject and remove embedding to avoid polluting future searches
        submission.status = Submission.REFUSED
        submission.flow_stage = Submission.FLOW_REFUSED
        submission.message = "Submission rejected due to redundancy"
        submission.save(update_fields=["status", "flow_stage", "message"])

        try:
            from qdrant_client.models import Filter as QFilter
            client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=QFilter(must=[FieldCondition(key="photo_id", match=MatchValue(value=int(photo.id)))]),
            )
        except Exception:
            pass

        return {"stop": True, "submission_id": submission_id, "user_id": user_id, "is_redundant": True}

    prev["is_redundant"] = False
    return prev

def kickoff_from_decoding(*, submission_id: int, user_id: int, reconstructed_path: str, description: str) -> str:
    res = chain(
        stage_relevance.s(submission_id=submission_id, user_id=user_id, reconstructed_path=reconstructed_path, description=description),
        stage_redundancy.s(),
    ).apply_async()
    return res.id


# -----------------------------
# Phase B: process original -> upload -> final accept notification
# -----------------------------
def _decode_base64_image_bytes(base64_image: str) -> bytes:
    if not base64_image:
        return b""
    payload = base64_image.split(",", 1)[1] if "," in base64_image else base64_image
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload") from exc


def _photo_embedding_from_bytes(raw: bytes) -> list[float]:
    img = Image.open(BytesIO(raw)).convert("RGB")
    inputs = clip_processor(
        text="prompts",
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=clip_processor.tokenizer.model_max_length,
    ).to(device)
    with __import__("torch").no_grad():
        vec = clip_model(**inputs).image_embeds[0].detach().cpu().numpy().astype("float64")
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Original photo embedding has zero norm")
    return (vec / norm).tolist()


def _cosine_similarity(a, b) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.shape != bv.shape or av.size == 0:
        raise ValueError("Embedding shape mismatch")
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an <= 0 or bn <= 0:
        raise ValueError("Cannot compare zero-norm embeddings")
    return float(np.dot(av, bv) / (an * bn))


def _delete_current_photo_embedding(photo_id: int) -> None:
    try:
        from qdrant_client.models import Filter as QFilter
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=QFilter(
                must=[FieldCondition(key="photo_id", match=MatchValue(value=int(photo_id)))]
            ),
        )
    except Exception:
        logger.exception("Failed to delete photo embedding photo_id=%s", photo_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def process_original_and_finalize(
    self,
    *,
    submission_id: int,
    user_id: int,
    base64_image: str,
    file_name: str,
    flow_nonce: str,
) -> Dict[str, Any]:
    """
    Phase B photo finalizer.

    Security invariants are checked again inside the Celery worker so a stale task,
    direct task invocation, replay, or view-layer regression cannot approve a submission
    that did not pass Phase A.
    """
    submission = Submission.objects.select_related(
        "event", "worker", "worker__user", "photo"
    ).get(id=submission_id)

    if submission.worker.user_id != int(user_id):
        return {"ok": False, "reason": "owner_mismatch"}
    if str(submission.flow_nonce) != str(flow_nonce):
        return {"ok": False, "reason": "stale_flow_nonce"}
    if submission.status != Submission.PENDING:
        return {"ok": False, "reason": f"invalid_status:{submission.status}"}
    if submission.flow_stage != Submission.FLOW_FINALIZING:
        return {"ok": False, "reason": f"invalid_flow_state:{submission.flow_stage}"}
    if not submission.photo_id:
        return {"ok": False, "reason": "missing_photo"}

    photo = submission.photo
    raw_original = _decode_base64_image_bytes(base64_image)
    if not raw_original:
        raise ValueError("Empty original image")

    # Bind Phase-B original to the Phase-A validated reconstruction.
    validated_embedding = scan_photo_embedding(photo.id)
    if not validated_embedding:
        raise RuntimeError("Validated photo embedding is missing from Qdrant")
    original_embedding = _photo_embedding_from_bytes(raw_original)
    binding_similarity = _cosine_similarity(validated_embedding, original_embedding)
    binding_threshold = configured_binding_threshold(
        "SUBMISSION_IMAGE_ORIGINAL_BINDING_MIN_COSINE"
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
                    message="Submission rejected: uploaded original does not match the validated photo.",
                )
        _delete_current_photo_embedding(photo.id)
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

    # Re-check capacity while the event row is locked. FINALIZING rows are reservations.
    with transaction.atomic():
        locked = (
            Submission.objects.select_for_update()
            .select_related("event", "worker", "worker__user", "photo")
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
            _delete_current_photo_embedding(photo.id)
            return {"ok": False, "reason": capacity_reason}

    # Blur and upload only after identity, state, binding and quota checks pass.
    blur_result = blur_faces_pipeline_from_base64(base64_image)
    image_bytes: bytes = blur_result["image_bytes"]

    s3_storage = CustomS3Boto3Storage(
        folder_name="submissions",
        dynamic_folder_name=f"{submission.event.title}/{submission.worker.user.fullName}/{submission.id}",
    )
    photo.image = s3_storage.save(file_name, ContentFile(image_bytes))
    photo.save(update_fields=["image"])

    try:
        original_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_img = ImageOps.exif_transpose(original_img)
        original_img.thumbnail((300, 300))
        thumb_io = BytesIO()
        original_img.save(thumb_io, format="JPEG", quality=60)
        thumb_s3_storage = CustomS3Boto3Storage(
            folder_name="thumbs",
            dynamic_folder_name=s3_storage.dynamic_folder_name,
        )
        thumb_filename = f"thumb_{file_name.rsplit('.', 1)[0]}.jpg"
        photo.thumbnail = thumb_s3_storage.save(thumb_filename, ContentFile(thumb_io.getvalue()))
        photo.save(update_fields=["thumbnail"])
    except Exception:
        logger.exception("Thumbnail generation failed submission_id=%s", submission_id)

    # Final compare-and-set. This is the only normal code path allowed to approve a photo.
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
            _delete_current_photo_embedding(photo.id)
            return {"ok": False, "reason": capacity_reason}

        locked.status = Submission.APPROVED
        locked.flow_stage = Submission.FLOW_FINALIZED
        locked.flow_finalized_at = timezone.now()
        locked.message = "Submission accepted"
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
        "[WS] photo finalized corr=%s submission_id=%s binding_similarity=%.5f",
        corr_id,
        submission_id,
        binding_similarity,
    )
    return {"ok": True, "binding_similarity": binding_similarity}

