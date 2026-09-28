import ast
import base64
import hashlib
import json
import os
import re
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from openai import OpenAI


ImageInput = Union[str, bytes, bytearray, np.ndarray]

COLOR_STRATEGY = "palette_by_label"

DEFAULT_TASK = "Detect and annotate all visible objects in this image."
DEFAULT_MAX_TOKENS = 900

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE_MB = 10

_LABEL_PALETTE_BGR: Tuple[Tuple[int, int, int], ...] = (
    (255, 128, 0),
    (0, 140, 255),
    (0, 200, 0),
    (255, 0, 255),
    (0, 220, 255),
    (255, 255, 0),
    (0, 0, 255),
    (180, 0, 180),
    (0, 165, 255),
    (203, 192, 255),
    (128, 128, 0),
    (0, 128, 128),
)

_COLOR_NAMES: Tuple[str, ...] = (
    "blue",
    "orange",
    "green",
    "magenta",
    "yellow",
    "cyan",
    "red",
    "purple",
    "amber",
    "pink",
    "olive",
    "teal",
)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int(
    value: Any,
    default: int,
    min_value: int = 100,
    max_value: int = 4000,
) -> int:
    try:
        value = int(value)
    except Exception:
        return default

    return max(min_value, min(value, max_value))


def validate_and_read_image(image: UploadedFile) -> bytes:
    if image is None:
        raise ValueError("Image is required.")

    max_size = MAX_IMAGE_SIZE_MB * 1024 * 1024

    if image.size > max_size:
        raise ValueError(f"Image too large. Maximum size is {MAX_IMAGE_SIZE_MB} MB.")

    try:
        image.seek(0)
    except Exception:
        pass

    image_bytes = image.read()

    if not image_bytes:
        raise ValueError("Empty image file.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if decoded is None:
        raise ValueError("Invalid or unsupported image file.")

    return image_bytes


def run_annotation(
    image: UploadedFile,
    task: str,
    max_tokens: int,
) -> Dict[str, Any]:
    image_bytes = validate_and_read_image(image)

    return annotate_image_openrouter(
        image_input=image_bytes,
        task=task,
        max_tokens=max_tokens,
    )

def _configured_annotation_model() -> str:
    """Primary annotation VLM from Django settings only."""
    model = str(
        getattr(settings, "OPENROUTER_ANNOTATION_MODEL", "") or ""
    ).strip()

    if not model:
        raise RuntimeError(
            "OPENROUTER_ANNOTATION_MODEL is not configured in Django settings."
        )

    return model


def _configured_annotation_fallback_models() -> List[str]:
    """Ordered annotation fallback models from Django settings only."""
    primary = _configured_annotation_model()
    raw = getattr(
        settings,
        "OPENROUTER_ANNOTATION_FALLBACK_MODELS",
        [],
    ) or []

    if isinstance(raw, str):
        raw = [raw]

    fallbacks: List[str] = []

    for value in raw:
        model = str(value or "").strip()

        if model and model != primary and model not in fallbacks:
            fallbacks.append(model)

    return fallbacks


def _configured_openrouter_keys_file() -> Path:
    """Central paid OpenRouter key file from Django settings."""
    raw = str(getattr(settings, "OPENROUTER_KEYS_FILE", "") or "").strip()

    if not raw:
        raise RuntimeError(
            "OPENROUTER_KEYS_FILE is not configured in Django settings."
        )

    path = Path(raw).expanduser()

    if not path.is_file():
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {path}")

    return path


_OPENROUTER_KEY_LOCK = Lock()
_OPENROUTER_KEY_INDEX = int.from_bytes(os.urandom(2), "big")


def _load_openrouter_keys() -> list[str]:
    """
    Load OpenRouter credentials exclusively from settings.OPENROUTER_KEYS_FILE.
    """
    key_file = _configured_openrouter_keys_file()
    keys: list[str] = []

    with key_file.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            key = raw_line.strip()

            if not key or key.startswith("#"):
                continue

            if key.lower().startswith("bearer "):
                key = key[7:].strip()

            if "=" in key:
                key = key.split("=", 1)[1].strip()

            if key and key not in keys:
                keys.append(key)

    if not keys:
        raise RuntimeError(
            f"No OpenRouter keys found in configured key file: {key_file}"
        )

    return keys


def resolve_openrouter_api_key() -> str:
    """
    Reads OpenRouter keys from settings.OPENROUTER_KEYS_FILE
    and rotates them in round-robin order.
    """

    global _OPENROUTER_KEY_INDEX

    keys = _load_openrouter_keys()

    _OPENROUTER_KEY_LOCK.acquire()
    try:
        key = keys[_OPENROUTER_KEY_INDEX % len(keys)]
        _OPENROUTER_KEY_INDEX += 1
    finally:
        _OPENROUTER_KEY_LOCK.release()

    return key.strip()


def classify_openrouter_retry_action(error: Exception) -> str:
    """
    Classifies OpenRouter/OpenAI client failures.

    Returns:
        - "next_token": retry same key with a smaller max_tokens value.
        - "next_key": rotate to another key.
        - "fatal": do not retry because the request/model is probably invalid.
    """

    msg = str(error).lower()

    token_markers = (
        "fewer max_tokens",
        "max_tokens is too large",
        "reduce max_tokens",
        "maximum context length",
        "context length",
    )

    if any(marker in msg for marker in token_markers):
        return "next_token"

    key_or_quota_markers = (
        "402",
        "credits",
        "insufficient credit",
        "insufficient credits",
        "quota",
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "401",
        "unauthorized",
        "invalid api key",
        "invalid_api_key",
        "api key",
        "temporarily unavailable",
        "overloaded",
        "timeout",
        "timed out",
        "connection error",
    )

    if any(marker in msg for marker in key_or_quota_markers):
        return "next_key"

    return "fatal"


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    raw = (text or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```\s*(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    def as_items(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]

        if isinstance(obj, dict):
            if any(
                key in obj
                for key in ("bbox", "box", "bounding_box", "boundingBox", "coordinates")
            ):
                return [obj]

            for key in ("detections", "objects", "annotations", "items", "results"):
                value = obj.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]

        return []

    def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique = []
        seen = set()

        for item in items:
            try:
                key = json.dumps(item, sort_keys=True, default=str)
            except Exception:
                key = repr(item)

            if key not in seen:
                unique.append(item)
                seen.add(key)

        return unique

    def loads_candidate(candidate: str) -> List[Dict[str, Any]]:
        candidate = candidate.strip().rstrip(",")

        if not candidate:
            return []

        for loader in (json.loads, ast.literal_eval):
            try:
                items = as_items(loader(candidate))
                if items:
                    return items
            except Exception:
                pass

        return []

    items = loads_candidate(raw)

    if items:
        return dedupe(items)

    decoder = json.JSONDecoder()
    decoded_items = []
    i = 0

    while i < len(raw):
        if raw[i] not in "[{":
            i += 1
            continue

        try:
            obj, end = decoder.raw_decode(raw[i:])
        except Exception:
            i += 1
            continue

        items = as_items(obj)

        if items:
            decoded_items.extend(items)
            i += max(end, 1)
        else:
            i += 1

    if decoded_items:
        return dedupe(decoded_items)

    recovered = []

    pattern = r"\{[^{}]*['\"](?:bbox|box|bounding_box|boundingBox|coordinates)['\"][^{}]*\}"

    for match in re.finditer(pattern, raw, re.DOTALL):
        items = loads_candidate(match.group(0))
        if items:
            recovered.extend(items)

    return dedupe(recovered)


def extract_bbox(item: Dict[str, Any]) -> Optional[List[float]]:
    bbox = (
        item.get("bbox")
        or item.get("box")
        or item.get("bounding_box")
        or item.get("boundingBox")
        or item.get("coordinates")
    )

    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x_min", "y_min", "x_max", "y_max")):
            bbox = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
        elif all(k in bbox for k in ("xmin", "ymin", "xmax", "ymax")):
            bbox = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        elif all(k in bbox for k in ("x", "y", "width", "height")):
            bbox = [
                bbox["x"],
                bbox["y"],
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            ]

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        return [float(v) for v in bbox]
    except Exception:
        return None


def bbox_to_pixels(bbox: List[float], width: int, height: int) -> Optional[List[int]]:
    x1, y1, x2, y2 = bbox
    max_coord = max(abs(v) for v in bbox)

    if max_coord <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_coord <= 1000:
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    x1_i = max(0, min(int(round(x1)), width - 1))
    y1_i = max(0, min(int(round(y1)), height - 1))
    x2_i = max(x1_i + 1, min(int(round(x2)), width))
    y2_i = max(y1_i + 1, min(int(round(y2)), height))

    if x2_i <= x1_i or y2_i <= y1_i:
        return None

    return [x1_i, y1_i, x2_i, y2_i]


def box_area(box: List[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


def box_iou(a: List[int], b: List[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = box_area([ix1, iy1, ix2, iy2])

    if inter <= 0:
        return 0.0

    union = box_area(a) + box_area(b) - inter

    return inter / union if union > 0 else 0.0


def contains_box(outer: List[int], inner: List[int], threshold: float = 0.82) -> bool:
    ix1 = max(outer[0], inner[0])
    iy1 = max(outer[1], inner[1])
    ix2 = min(outer[2], inner[2])
    iy2 = min(outer[3], inner[3])

    inter = box_area([ix1, iy1, ix2, iy2])
    inner_area = box_area(inner)

    return inner_area > 0 and (inter / inner_area) >= threshold


def dedupe_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []

    for det in detections:
        label_key = normalize_label(str(det.get("label") or "object"))
        box = det["bbox"]
        duplicate_idx = None

        for idx, existing in enumerate(unique):
            same_label = label_key == normalize_label(str(existing.get("label") or "object"))

            if not same_label:
                continue

            existing_box = existing["bbox"]

            if box == existing_box or box_iou(box, existing_box) >= 0.92:
                duplicate_idx = idx
                break

            if contains_box(existing_box, box, threshold=0.94) or contains_box(
                box,
                existing_box,
                threshold=0.94,
            ):
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            unique.append(det)
        else:
            current_area = box_area(box)
            kept_area = box_area(unique[duplicate_idx]["bbox"])

            if 0 < current_area < kept_area:
                unique[duplicate_idx] = det

    return unique


def is_broad_scene_label(label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()

    broad_terms = {
        "background",
        "image",
        "scene",
        "street scene",
        "graffiti",
        "graffiti wall",
        "mural",
        "wall",
        "building facade",
        "facade",
        "sky",
        "ground",
        "road",
        "sidewalk",
        "pavement",
    }

    return normalized in broad_terms


def filter_detections(
    detections: List[Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    if len(detections) <= 1:
        return detections

    image_area = max(1, width * height)
    filtered = []

    for det in detections:
        box = det["bbox"]
        area_frac = box_area(box) / image_area

        contained_count = sum(
            1
            for other in detections
            if other is not det
            and box_area(other["bbox"]) < box_area(box)
            and contains_box(box, other["bbox"])
        )

        spans_scene = (
            area_frac >= 0.55
            or (
                (box[2] - box[0]) / max(1, width) >= 0.9
                and (box[3] - box[1]) / max(1, height) >= 0.6
            )
        )

        if spans_scene and (is_broad_scene_label(det["label"]) or contained_count >= 3):
            continue

        filtered.append(det)

    return filtered


def decode_image_input(image_input: ImageInput) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        arr = image_input

        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("Invalid numpy image shape. Expected HxWx3.")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

    if isinstance(image_input, (bytes, bytearray)):
        np_buffer = np.frombuffer(bytes(image_input), dtype=np.uint8)
        image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("Invalid image bytes.")

        return image_bgr

    if isinstance(image_input, str):
        image_bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError(f"Invalid image path or unreadable file: {image_input}")

        return image_bgr

    raise TypeError("Unsupported image input type.")


def bgr_to_rgb(bgr: Tuple[int, int, int]) -> List[int]:
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


def rgb_to_hex(rgb: List[int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def build_label_color_map(labels: List[str]) -> Dict[str, Dict[str, Any]]:
    ordered = []
    seen = set()

    for label in labels:
        norm = normalize_label(label)

        if norm in seen:
            continue

        seen.add(norm)
        ordered.append((norm, (label or "object").strip() or "object"))

    ordered.sort(key=lambda pair: pair[0])

    mapping = {}

    for idx, (norm, display_label) in enumerate(ordered):
        bgr = _LABEL_PALETTE_BGR[idx % len(_LABEL_PALETTE_BGR)]
        rgb = bgr_to_rgb(bgr)

        mapping[norm] = {
            "label": display_label,
            "color_bgr": list(bgr),
            "color_rgb": rgb,
            "color_hex": rgb_to_hex(rgb),
            "color_name": _COLOR_NAMES[idx % len(_COLOR_NAMES)],
            "palette_index": idx,
        }

    return mapping


def apply_detection_colors(
    detections: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    label_color_map = build_label_color_map(
        [str(d.get("label") or "object") for d in detections]
    )

    enriched = []

    for idx, det in enumerate(detections, start=1):
        label = str(det.get("label") or "object")
        norm = normalize_label(label)
        colors = label_color_map[norm]

        enriched.append(
            {
                "id": idx,
                "label": label,
                "bbox": det["bbox"],
                "color_rgb": colors["color_rgb"],
                "color_hex": colors["color_hex"],
                "color_name": colors["color_name"],
                "palette_index": colors["palette_index"],
            }
        )

    legend = []

    for norm, colors in sorted(
        label_color_map.items(),
        key=lambda item: item[1]["palette_index"],
    ):
        instance_count = sum(
            1 for det in enriched if normalize_label(det["label"]) == norm
        )

        legend.append(
            {
                "label": colors["label"],
                "color_rgb": colors["color_rgb"],
                "color_hex": colors["color_hex"],
                "color_name": colors["color_name"],
                "palette_index": colors["palette_index"],
                "instance_count": instance_count,
            }
        )

    return enriched, legend


def draw_labeled_box(image_bgr: np.ndarray, det: Dict[str, Any]) -> None:
    height, width = image_bgr.shape[:2]

    label = str(det.get("label") or "object")
    x1, y1, x2, y2 = det["bbox"]

    color_rgb = det["color_rgb"]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, 2)

    (label_width, label_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1,
    )

    label_x2 = min(width - 1, x1 + label_width + 8)
    label_y1 = max(0, y1 - label_height - baseline - 8)

    cv2.rectangle(image_bgr, (x1, label_y1), (label_x2, y1), color_bgr, -1)

    cv2.putText(
        image_bgr,
        label,
        (x1 + 4, max(label_height + 2, y1 - baseline - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def format_detections_summary(result: Dict[str, Any]) -> str:
    detections = result.get("detections") or []
    raw_count = int(result.get("raw_model_item_count") or 0)

    if not detections:
        return "No detections."

    lines = [
        f"Parsed raw model items: {raw_count}; drawn valid boxes: {len(detections)}",
        f"Color strategy: {result.get('color_strategy', COLOR_STRATEGY)}",
        "",
        "Legend:",
    ]

    for entry in result.get("label_colors") or []:
        lines.append(
            f"  • {entry.get('label')} -> {entry.get('color_name')} "
            f"{entry.get('color_hex')} rgb={entry.get('color_rgb')}"
        )

    lines.append("")
    lines.append("Detections:")

    for det in detections:
        lines.append(
            f"#{det.get('id')} label={det.get('label')} "
            f"color={det.get('color_name')} {det.get('color_hex')} "
            f"bbox={det.get('bbox')}"
        )

    return "\n".join(lines)


def error_result(message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "color_strategy": COLOR_STRATEGY,
        "raw_model_item_count": 0,
        "label_colors": [],
        "detections": [],
        "message": message,
    }


def annotation_payload_for_export(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "status": result.get("status"),
        "color_strategy": result.get("color_strategy", COLOR_STRATEGY),
        "raw_model_item_count": result.get("raw_model_item_count", 0),
        "label_colors": result.get("label_colors") or [],
        "detections": result.get("detections") or [],
        "model": result.get("model"),
        "requested_model": result.get("requested_model"),
        "configured_fallback_models": result.get(
            "configured_fallback_models"
        ) or [],
    }

    if result.get("status") != "success":
        payload["message"] = result.get("message")

    return payload


def annotate_image_openrouter(
    image_input: ImageInput,
    task: str = DEFAULT_TASK,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    try:
        primary_model = _configured_annotation_model()
        fallback_models = _configured_annotation_fallback_models()
        model_chain = [primary_model, *fallback_models]

        frame_bgr = decode_image_input(image_input)

        ok, encoded = cv2.imencode(".jpg", frame_bgr)

        if not ok:
            return error_result("Failed to encode input image.")

        frame_bytes = encoded.tobytes()

        data_url = "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode("utf-8")

        system_prompt = (
            "You are a precise visual annotation engine. "
            "First produce the raw object list from the image, then assign one tight bounding box to every "
            "visible object or meaningful region in that raw list. Include partially visible objects, readable "
            "text regions, people/animals, vehicles, plants, tools, cables, furniture, surfaces, walls, and "
            "background regions when they are visible. Do not drop an object because it contains smaller "
            "objects or because it is a broad region. "
            "Return ONLY a valid JSON array. "
            'Each item must be {"label":"short class name","bbox":[x_min,y_min,x_max,y_max]}. '
            "Bounding boxes must be normalized on 0..1000 coordinates and should cover the full visible extent."
        )

        user_text = (
            f"Task: {(task or '').strip() or 'Detect and annotate visible objects.'}\n"
            "Output strictly valid JSON array only."
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ]

        attempts = []

        for token_count in [int(max_tokens), 900, 700, 500, 350, 220]:
            if token_count not in attempts:
                attempts.append(token_count)

        response = None
        last_error = None
        keys = _load_openrouter_keys()
        tried_key_count = 0

        # Try each configured key. For each key, also try smaller max_tokens
        # values only when OpenRouter says the token budget is too high.
        for _ in range(len(keys)):
            api_key = resolve_openrouter_api_key()
            tried_key_count += 1

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            rotate_key = False

            for token_count in attempts:
                try:
                    response = client.chat.completions.create(
                        model=primary_model,
                        messages=messages,
                        max_tokens=token_count,
                        temperature=0.1,
                        top_p=0.9,
                        extra_body={
                            "models": model_chain,
                            "provider": {
                                "allow_fallbacks": True,
                            },
                        },
                    )
                    last_error = None
                    break

                except Exception as e:
                    last_error = e
                    retry_action = classify_openrouter_retry_action(e)

                    if retry_action == "next_token":
                        continue

                    if retry_action == "next_key":
                        rotate_key = True
                        break

                    return error_result(str(e))

            if response is not None:
                break

            if not rotate_key and last_error is not None:
                break

        if response is None:
            return error_result(
                f"Annotation failed after trying {tried_key_count} key(s) "
                f"with token fallbacks {attempts}: {last_error}"
            )

        raw_output = response.choices[0].message.content if response.choices else ""

        items = extract_json_array(raw_output)

        height, width = frame_bgr.shape[:2]
        detections = []
        annotated = frame_bgr.copy()

        for item in items:
            bbox = extract_bbox(item)

            label = str(
                item.get("label")
                or item.get("class")
                or item.get("name")
                or "object"
            ).strip() or "object"

            if bbox is None:
                continue

            pixel_bbox = bbox_to_pixels(bbox, width, height)

            if pixel_bbox is None:
                continue

            detections.append(
                {
                    "label": label,
                    "bbox": pixel_bbox,
                }
            )

        detections = dedupe_detections(detections)
        detections = filter_detections(detections, width, height)
        detections, label_colors = apply_detection_colors(detections)

        for det in detections:
            draw_labeled_box(annotated, det)

        ok, output_encoded = cv2.imencode(".jpg", annotated)

        if not ok:
            return error_result("Failed to encode annotated image.")

        output_bytes = output_encoded.tobytes()

        result = {
            "status": "success",
            "color_strategy": COLOR_STRATEGY,
            "raw_model_item_count": len(items),
            "label_colors": label_colors,
            "detections": detections,
            "annotated_image_base64": base64.b64encode(output_bytes).decode("utf-8"),
            "raw_model_output": raw_output or "",
            "model": (
                str(getattr(response, "model", "") or "").strip()
                or primary_model
            ),
            "requested_model": primary_model,
            "configured_fallback_models": fallback_models,
        }

        result["detections_summary"] = format_detections_summary(result)

        return result

    except Exception as e:
        return error_result(str(e))


def build_annotation_response(
    result: Dict[str, Any],
    include_base64: bool = True,
    include_summary: bool = True,
    include_raw_model_output: bool = False,
) -> Dict[str, Any]:
    payload = annotation_payload_for_export(result)

    if include_base64:
        payload["annotated_image_base64"] = result.get("annotated_image_base64")

    if include_summary:
        payload["detections_summary"] = result.get("detections_summary") or format_detections_summary(result)

    if include_raw_model_output:
        payload["raw_model_output"] = result.get("raw_model_output", "")

    return payload