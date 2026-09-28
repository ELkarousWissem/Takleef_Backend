import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import radians, cos, sin, atan2, sqrt
from pathlib import Path

import cv2
import nltk
import numpy as np
import torch
from PIL import Image
from celery import shared_task
from nltk.corpus import stopwords
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from rest_framework import status, permissions
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from transformers import XCLIPProcessor, XCLIPModel
from transformers import AutoProcessor, AutoModel

from django.conf import settings
from mobicrowd.models.FileStorage import CustomS3Boto3Storage
from mobicrowd.models.Users import Worker
from mobicrowd.models.submisson import Submission, Video, Event, WorkerReward, UserUploadLog
from mobicrowd.tasks import BASE_DIR, logger
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ─── NLTK setup ──────────────────────────────
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ─── CLIP setup ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xclip_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
xclip_model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot")

def clean_text(text: str) -> str:
    tokens = text.lower().split()
    return " ".join([t for t in tokens if t not in STOP_WORDS])

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
def run_relevance_check(frames, description, submission_id, threshold=0.75):
    try:
        cleaned_desc = clean_text(description)
        positive_text = cleaned_desc
        negative_text = "irrelevant, this is not relevant to the task"
        negatives = [p.strip() for p in negative_text.split(",") if p.strip()]
        submission = Submission.objects.get(id=submission_id)
        video_id = submission.video.id
        video = submission.video
        xclip_size = (224, 224)
        resized_frames = [cv2.resize(f, xclip_size) for f in frames]
        pil_frames = [Image.fromarray(f) for f in resized_frames]
        inputs = xclip_processor(
            text=[positive_text]+ negatives,
            videos=[pil_frames],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = xclip_model(**inputs)
            probs = outputs.logits_per_video.softmax(dim=1)
            score = probs[0, 0].item()
            video_embeds = torch.nn.functional.normalize(outputs.video_embeds, p=2, dim=-1)
            print(f"- XCLIP relevance score: {score:.3f}")
            is_relevant = score >= threshold

        if is_relevant:
            # Extract normalized image embedding
            embedding = video_embeds.cpu().numpy().squeeze().flatten().tolist()
            payload = {
                "video_id": video_id,
                "submission_id": submission.id,
                "event_id": submission.event.id,
                "metadata": video.metadata
            }
            point = PointStruct(
                id=int(submission.id),
                vector=embedding,
                payload=payload
            )
            client.upsert(
                collection_name=settings.QDRANT_COLLECTION_VIDEO,
                points=[point]
            )
        else:
            submission.status = submission.REFUSED
            submission.message="Submission rejected due to irrelevance"
            submission.save()
            print("not relevant")
        return {
            "relevance_score": round(score, 4),
            "is_relevant": is_relevant
        }

    except Exception as e:
        return {"error": str(e)}
def _num_key(p: Path) -> int:
    try:
        return int(p.stem)
    except Exception:
        return 10**9

def _load_rgb_resize(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    """Read with OpenCV, convert BGR->RGB, resize, return np.ndarray(H,W,3) or None."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if size:
        img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    return img_rgb
SAFE_TMP_ROOT = (BASE_DIR / "tmp" / "reconstructed videos").resolve()

def _safe_delete_dir(p: Path) -> None:
    """Delete a directory only if it’s under SAFE_TMP_ROOT; swallow errors as warnings."""
    p = p.resolve()
    try:
        # ensure p is inside SAFE_TMP_ROOT
        p.relative_to(SAFE_TMP_ROOT)
    except ValueError:
        logger.warning("Skip deletion outside safe root: %s", p)
        return
    if p.is_dir():
        shutil.rmtree(p)
class RelevanceFromFramesFolderView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        return Response(
            {
                "code": "legacy_submission_endpoint_disabled",
                "message": "Use the server-authoritative /submissions/video/kickoff/<submission_id> flow.",
            },
            status=status.HTTP_410_GONE,
        )

def scan_video_embedding(video_id: int):
    """
    Fallback method to retrieve the embedding vector for a photo using a brute-force scan,
    when indexed filtering fails. Works reliably as long as photo_id is unique.
    """
    try:
        scroll_result, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_VIDEO,
            with_vectors=True,
            with_payload=True,
            limit=1000  # you can increase this if needed
        )

        for point in scroll_result:
            if point.payload.get("video_id") == video_id:
                return point.vector

        logging.warning(f"⚠️ Photo ID {video_id} not found in Qdrant collection.")
        return None

    except Exception as e:
        logging.error(f"❌ Error during scan for photo {video_id}: {e}")
        return None


from datetime import datetime, timezone
from math import radians, sin, cos, atan2, sqrt

DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"  # kept for backward compatibility

def haversine(coord1, coord2):
    """Return distance in kilometers between two (lat, lon) pairs."""
    R = 6371.0  # earth radius in km
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------- helpers: robust timestamp + GPS extraction ----------
def _parse_dt_any(x):
    """
    Accepts:
      - ISO 8601, e.g. "2025-11-01T09:31:54.885Z" (like your screenshot)
      - 'YYYY:MM:DD HH:MM:SS' per DATETIME_FORMAT
      - unix seconds (int/float)
    Returns timezone-aware UTC datetime or None.
    """
    if not x:
        return None
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    if isinstance(x, str):
        s = x.strip()
        # Try ISO-8601
        try:
            if s.endswith("Z"):
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            return datetime.fromisoformat(s)
        except Exception:
            pass
        # Try legacy format
        try:
            return datetime.strptime(s, DATETIME_FORMAT).replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _extract_lat_lon_dt(meta: dict):
    """
    Expects the shape in your screenshot:
      meta = {
        "media": {...},
        "capture": {
          "gps": {"source": "...", "latitude": 34.7485, "longitude": 10.72067},
          "type": "video",
          "timestamp": "2025-11-01T09:31:54.885Z"
        }
      }
    Fallbacks supported:
      - top-level 'lat'/'lon' or 'latitude'/'longitude'
      - 'datetime' or 'timestamp' at top-level or in capture
    """
    lat = lon = dt = None
    if isinstance(meta, dict):
        cap = meta.get("capture") or {}
        gps = cap.get("gps") or {}

        # latitude / longitude (preferred nested)
        for k in ("latitude", "lat"):
            if gps.get(k) is not None:
                try:
                    lat = float(gps[k])
                except Exception:
                    pass
                break
        for k in ("longitude", "lon"):
            if gps.get(k) is not None:
                try:
                    lon = float(gps[k])
                except Exception:
                    pass
                break

        # fallbacks (top-level)
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


# ---------- main task ----------
@shared_task
def check_redundancy_task(
    submission_id,
    threshold: float = 0.75,
    location_threshold: float = 0.05,  # kilometers (50 m)
    time_threshold: int = 86400        # seconds (24 h)
):
    """
    Redundancy criteria:
      1) a similar point in Qdrant with score >= threshold AND different video_id
      2) that similar point is within 'location_threshold' km of current GPS
      3) and the time difference <= 'time_threshold' seconds
    If redundant => REFUSED and (optionally) remove the current video's point from Qdrant.
    """
    try:
        submission = Submission.objects.get(id=submission_id)
        video = submission.video
        metadata = video.metadata or {}

        # current item's geo/time
        cur_lat, cur_lon, cur_dt = _extract_lat_lon_dt(metadata)
        print(cur_lat, cur_lon , cur_dt)

        # We can still proceed if timestamp missing, but geo is required for the new logic
        if cur_lat is None or cur_lon is None:
            return {"error": "Missing latitude/longitude in current video's metadata."}
        if cur_dt is None:
            # best-effort: treat 'now' as current time so time checks still work
            cur_dt = datetime.now(timezone.utc)

        # vector search
        embedding = scan_video_embedding(video.id)
        if not embedding:
            return {"error": "Embedding not found in Qdrant."}

        results = client.search(
            collection_name=settings.QDRANT_COLLECTION_VIDEO,
            query_vector=embedding,
            limit=10,
            query_filter=Filter(
                must=[FieldCondition(key="event_id", match=MatchValue(value=int(submission.event.id)))]
            )
        )
        print(results)

        # similarity + not the same video
        similar = [
            p for p in results
            if float(p.score) >= threshold and str(p.payload.get("video_id")) != str(video.id)
        ]

        # geo/time filtering
        nearby = []
        all_matches_out = []  # for returning diagnostics
        for p in similar:
            p_meta = (p.payload or {}).get("metadata", {})
            plat, plon, pdt = _extract_lat_lon_dt(p_meta)

            # compute diagnostics even if something is missing
            diag = {
                **(p.payload or {}),
                "similarity_score": round(float(p.score), 4),
                "distance_km": None,
                "time_delta_sec": None,
            }

            if plat is not None and plon is not None:
                dist_km = haversine((cur_lat, cur_lon), (plat, plon))
                diag["distance_km"] = round(dist_km, 6)
            else:
                dist_km = None

            if pdt is not None and cur_dt is not None:
                td = abs((cur_dt - pdt).total_seconds())
                diag["time_delta_sec"] = round(td, 3)
            else:
                td = None

            all_matches_out.append(diag)

            if (dist_km is not None and dist_km <= location_threshold and
                td is not None and td <= time_threshold):
                # candidate that is both close in space and time
                nearby.append((p, dist_km, td, pdt))

        # decide redundancy
        is_redundant = False
        distance_to_match = None
        time_diff = None
        most_recent_point = None

        if nearby:
            # pick the most recent among the nearby spatiotemporal matches
            most_recent_point, distance_to_match, time_diff, _ = max(
                nearby, key=lambda t: (t[3] or datetime.fromtimestamp(0, tz=timezone.utc))
            )
            is_redundant = True

            submission.status = submission.REFUSED
            if hasattr(submission, "flow_stage"):
                submission.flow_stage = Submission.FLOW_REFUSED
                submission.message = "Submission rejected due to redundancy"
                submission.save(update_fields=["status", "flow_stage", "message"])
            else:
                submission.message = "Submission rejected due to redundancy"
                submission.save(update_fields=["status", "message"])

            # keep the collection clean by removing the *current* video's point
            try:
                client.delete(
                    collection_name=settings.QDRANT_COLLECTION_VIDEO,
                    points_selector=Filter(
                        must=[FieldCondition(key="video_id", match=MatchValue(value=int(video.id)))]
                    )
                )
            except Exception as qerr:
                print(f"⚠️ Failed to delete current video {video.id} from Qdrant: {qerr}")
        else:
            # Legacy redundancy task is non-authoritative. It may report PASS but
            # must never approve; approval belongs only to the secured Phase-B finalizer.
            pass

        # shape "most_recent_match" like before, if we have one
        most_recent_out = None
        if most_recent_point is not None:
            most_recent_out = {
                **(most_recent_point.payload or {}),
                "similarity_score": round(float(most_recent_point.score), 4)
            }

        return {
            "is_redundant": is_redundant,
            "distance_to_match": round(distance_to_match, 6) if distance_to_match is not None else None,
            "time_diff_sec": round(time_diff, 3) if time_diff is not None else None,
            "location_threshold_km": location_threshold,
            "time_threshold_sec": time_threshold,
            "matches": all_matches_out,
            "most_recent_match": most_recent_out
        }

    except Submission.DoesNotExist:
        return {"error": "Submission not found."}
    except Exception as e:
        return {"error": str(e)}
class RedundancyCheckVideoWithMetadataView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        return Response(
            {
                "code": "legacy_submission_endpoint_disabled",
                "message": "Video redundancy is part of the authoritative kickoff flow and cannot be called independently.",
            },
            status=status.HTTP_410_GONE,
        )


from datetime import datetime
import os, io, base64, tempfile, subprocess, uuid
import cv2
from pathlib import Path
from django.core.files.base import ContentFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO
from PIL import Image, ImageOps



# ---------------- YOLO model loader ----------------
_yolo = None
YOLO_MODEL_PATH = "mobicrowd/models/UNIQUE/yolov8n-face.pt"

def get_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLO(YOLO_MODEL_PATH)
        try:
            _yolo.to("cuda")
        except Exception:
            pass
    return _yolo


# ---------------- Helper functions ----------------
def _expand_box(x1, y1, x2, y2, W, H, margin=0.25):
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * margin), int(bh * margin)
    nx1 = max(0, x1 - dx); ny1 = max(0, y1 - dy)
    nx2 = min(W - 1, x2 + dx); ny2 = min(H - 1, y2 + dy)
    return nx1, ny1, nx2, ny2

def _blur_boxes_bgr(frame_bgr, boxes_xyxy, ksize=31):
    out = frame_bgr
    k = ksize if ksize % 2 == 1 else ksize + 1
    for (x1, y1, x2, y2) in boxes_xyxy:
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        if roi.size:
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out

def _detect_faces_xyxy(model, frame_bgr, conf=0.25, imgsz=640):
    res = model.predict(frame_bgr, conf=conf, imgsz=imgsz, verbose=False)
    boxes = []
    if res and res[0].boxes and res[0].boxes.xyxy is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist()
    return boxes


# ---------------- Core blurring function ----------------
def blur_video_faces_sparse(
    in_path: str,
    out_path: str,
    stride: int = 1,
    margin: float = 0.25,
    conf: float = 0.25,
    imgsz: int = 640,
    blur_ksize: int = 31,
    writer_fourcc: str = "mp4v"
):
    model = get_yolo()
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*writer_fourcc)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise ValueError("VideoWriter failed — codec not available?")

    last_boxes = []
    det_frames = 0
    frames_written = 0
    thumb_jpg = None
    mid_idx = max(0, N // 2)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_det = (idx % stride == 0) or (not last_boxes)
        if run_det:
            raw = _detect_faces_xyxy(model, frame, conf=conf, imgsz=imgsz)
            boxes = []
            for (x1, y1, x2, y2) in raw:
                bx1, by1, bx2, by2 = _expand_box(int(x1), int(y1), int(x2), int(y2), W, H, margin=margin)
                boxes.append((bx1, by1, bx2, by2))
            last_boxes = boxes
            det_frames += 1

        blurred = _blur_boxes_bgr(frame, last_boxes, blur_ksize)
        writer.write(blurred)
        frames_written += 1

        if thumb_jpg is None and idx >= mid_idx:
            rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).convert("RGB")
            img = ImageOps.exif_transpose(img)
            buff = io.BytesIO()
            img.save(buff, format="JPEG", quality=75)
            thumb_jpg = buff.getvalue()

        idx += 1

    cap.release()
    writer.release()
    if thumb_jpg is None:
        thumb_jpg = b""

    return {
        "frames": frames_written,
        "detect_calls": det_frames,
        "thumbnail_jpg": thumb_jpg
    }


# ---------------- API View ----------------
class ProcessVideoBlurAndUploadView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        return Response(
            {
                "code": "legacy_submission_endpoint_disabled",
                "message": "Direct video processing/approval is disabled. Use /submissions/video/upload-original/<submission_id> with a valid finalize token.",
            },
            status=status.HTTP_410_GONE,
        )
