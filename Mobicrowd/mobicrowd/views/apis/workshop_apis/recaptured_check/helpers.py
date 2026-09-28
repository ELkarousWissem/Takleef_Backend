import os
import random
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import tensorflow as tf
from django.core.files.uploadedfile import UploadedFile


ImageInput = Union[str, bytes, bytearray, np.ndarray]

_RECAPTURE_MODEL = None
_MODEL_LOCK = Lock()

IMAGE_MAX_SIZE_MB = 15
VIDEO_MAX_SIZE_MB = 150

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """
    Compatibility layer for old/new Keras H5 configs.
    Some saved configs include unsupported keys like `groups`.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def get_recapture_model():
    global _RECAPTURE_MODEL

    if _RECAPTURE_MODEL is not None:
        return _RECAPTURE_MODEL

    _MODEL_LOCK.acquire()

    try:
        if _RECAPTURE_MODEL is None:
            model_path = Path("mobicrowd/models/UNIQUE/Final_Recaptured_model_Pruned.h5")

            if not model_path.exists():
                raise FileNotFoundError(f"Recaptured model not found: {model_path}")

            try:
                _RECAPTURE_MODEL = tf.keras.models.load_model(
                    str(model_path),
                    compile=False,
                )
            except Exception:
                _RECAPTURE_MODEL = tf.keras.models.load_model(
                    str(model_path),
                    compile=False,
                    custom_objects={
                        "DepthwiseConv2D": CompatibleDepthwiseConv2D,
                    },
                )

    finally:
        _MODEL_LOCK.release()

    return _RECAPTURE_MODEL


def laplacian_filter_rgb(src: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    def apply_filter(image: np.ndarray) -> np.ndarray:
        filtered_channels = []

        for channel_index in range(3):
            channel = image[:, :, channel_index]

            dst16 = cv2.filter2D(
                channel,
                cv2.CV_16S,
                kernel,
            )

            min_value = float(dst16.min())
            max_value = float(dst16.max())

            if max_value != min_value:
                scaled = (dst16 - min_value) / (max_value - min_value) * 255.0
            else:
                scaled = np.zeros_like(dst16)

            filtered_channels.append(scaled.astype(np.uint8))

        return cv2.merge(filtered_channels)

    lap1 = apply_filter(src)

    lap1_resized = cv2.resize(
        lap1,
        (512, 512),
        interpolation=cv2.INTER_LINEAR,
    )

    return apply_filter(lap1_resized)


def predict_recapture_score_from_rgb(frame_rgb: np.ndarray) -> float:
    model = get_recapture_model()

    filtered = laplacian_filter_rgb(frame_rgb)

    tensor = tf.expand_dims(
        tf.convert_to_tensor(filtered, dtype=tf.float32),
        axis=0,
    )

    prediction = model(tensor, training=False)

    score = float(prediction.numpy().squeeze())

    return max(0.0, min(score, 1.0))


def validate_and_read_uploaded_file(
    uploaded_file: UploadedFile,
    max_size_mb: int,
) -> bytes:
    if uploaded_file is None:
        raise ValueError("File is required.")

    max_size = max_size_mb * 1024 * 1024

    if uploaded_file.size > max_size:
        raise ValueError(f"File too large. Maximum size is {max_size_mb} MB.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("Empty file.")

    return file_bytes


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)

    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Invalid or unsupported image file.")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def check_recaptured_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    frame_rgb = decode_image_bytes(image_bytes)

    score = predict_recapture_score_from_rgb(frame_rgb)

    result = "recaptured" if score >= 0.5 else "original"

    return {
        "status": "success",
        "file_type": "image",
        "decision_threshold": 0.5,
        "recapture_score": score,
        "recapture_result": result,
    }


def sample_8_frame_indices(total_frames: int) -> List[int]:
    if total_frames <= 0:
        return []

    number_of_frames = min(8, total_frames)

    return sorted(random.sample(range(total_frames), number_of_frames))


def save_uploaded_video_to_temp(uploaded_file: UploadedFile) -> str:
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()

    if suffix not in VIDEO_EXTENSIONS:
        suffix = ".mp4"

    video_bytes = validate_and_read_uploaded_file(
        uploaded_file,
        max_size_mb=VIDEO_MAX_SIZE_MB,
    )

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:
        temp_file.write(video_bytes)
        return temp_file.name


def check_recaptured_video_path(video_path: str) -> Dict[str, Any]:
    if not video_path:
        return {
            "status": "error",
            "message": "Video path is required.",
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "status": "error",
            "message": "Failed to open video.",
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if fps <= 0:
            fps = 25.0

        indices = sample_8_frame_indices(total_frames)

        if not indices:
            return {
                "status": "error",
                "message": "No frames available in video.",
            }

        frame_scores = []
        recaptured_count = 0

        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ok, frame_bgr = cap.read()

            if not ok or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            score = predict_recapture_score_from_rgb(frame_rgb)
            is_recaptured = score >= 0.5

            if is_recaptured:
                recaptured_count += 1

            frame_scores.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(frame_index / fps, 3),
                    "score": round(score, 6),
                    "is_recaptured": bool(is_recaptured),
                }
            )

        checked = len(frame_scores)

        if checked == 0:
            return {
                "status": "error",
                "message": "Could not read sampled frames.",
            }

        ratio = recaptured_count / checked
        result = "recaptured" if ratio >= 0.5 else "original"

        return {
            "status": "success",
            "file_type": "video",
            "sampled_frames": checked,
            "sample_indices": [item["frame_index"] for item in frame_scores],
            "decision_rule": "video is recaptured if recaptured_ratio >= 0.50",
            "recaptured_frames": recaptured_count,
            "recaptured_ratio": round(ratio, 6),
            "recapture_result": result,
            "frame_scores": frame_scores,
        }

    finally:
        cap.release()


def infer_uploaded_file_type(
    uploaded_file: UploadedFile,
    requested_type: str = "",
) -> str:
    if uploaded_file is None:
        raise ValueError("File is required.")

    requested_type = str(requested_type or "").strip().lower()

    if requested_type in {"image", "video"}:
        return requested_type

    filename = getattr(uploaded_file, "name", "")
    suffix = Path(filename).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return "image"

    if suffix in VIDEO_EXTENSIONS:
        return "video"

    content_type = str(getattr(uploaded_file, "content_type", "") or "").lower()

    if content_type.startswith("image/"):
        return "image"

    if content_type.startswith("video/"):
        return "video"

    raise ValueError("Could not infer file type. Send media_type as image or video.")


def run_recaptured_check(
    uploaded_file: UploadedFile,
    media_type: str = "",
) -> Dict[str, Any]:
    detected_type = infer_uploaded_file_type(
        uploaded_file,
        requested_type=media_type,
    )

    if detected_type == "image":
        image_bytes = validate_and_read_uploaded_file(
            uploaded_file,
            max_size_mb=IMAGE_MAX_SIZE_MB,
        )

        result = check_recaptured_image_bytes(image_bytes)

        result["filename"] = getattr(uploaded_file, "name", "image")

        return result

    if detected_type == "video":
        temp_video_path = save_uploaded_video_to_temp(uploaded_file)

        try:
            result = check_recaptured_video_path(temp_video_path)

            result["filename"] = getattr(uploaded_file, "name", "video")

            return result

        finally:
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

    raise ValueError("Unsupported media type.")