import base64
import gc
import os
from io import BytesIO
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from django.core.files.uploadedfile import UploadedFile
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import Sam3Model, Sam3Processor


USE_CUDA = torch.cuda.is_available() and os.environ.get("SAM3_CPU", "").strip().lower() not in {
    "1",
    "true",
    "yes",
}

DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

_MODEL_CACHE: Optional[Tuple[Sam3Model, Sam3Processor]] = None
_MODEL_LOCK = Lock()

COLOR_STRATEGY = "palette_by_label"

MAX_IMAGE_SIZE_MB = 10

DEFAULT_THRESHOLD = 0.3
DEFAULT_MASK_THRESHOLD = 0.3

PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (0, 200, 0),
]

COLOR_NAMES = (
    "red",
    "green",
    "blue",
    "orange",
    "magenta",
    "cyan",
    "purple",
    "lime",
)

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_float(
    value: Any,
    default: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default

    return max(min_value, min(parsed, max_value))


def device_info() -> Dict[str, str]:
    return {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "hf_token": "set" if HF_TOKEN else "not set",
    }


def from_pretrained_kwargs() -> Dict[str, str]:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def get_sam3_image_model() -> Tuple[Sam3Model, Sam3Processor]:
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    _MODEL_LOCK.acquire()

    try:
        if _MODEL_CACHE is None:
            kwargs = from_pretrained_kwargs()

            model = Sam3Model.from_pretrained(
                "facebook/sam3",
                **kwargs,
            ).to(DEVICE)

            processor = Sam3Processor.from_pretrained(
                "facebook/sam3",
                **kwargs,
            )

            model.eval()

            _MODEL_CACHE = (model, processor)
    finally:
        _MODEL_LOCK.release()

    return _MODEL_CACHE


def unload_sam3_image_model() -> None:
    global _MODEL_CACHE

    _MODEL_CACHE = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    class DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    return DummyContext()


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

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        raise ValueError("Invalid or unsupported image file.")

    return image_bytes


def load_uploaded_image(image: UploadedFile) -> Image.Image:
    image_bytes = validate_and_read_image(image)

    try:
        pil_image = Image.open(BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)
        return pil_image.convert("RGB")
    except Exception:
        raise ValueError("Invalid or unsupported image file.")


def normalize_text_prompts(text_prompt: Any) -> List[str]:
    if text_prompt is None:
        return []

    if isinstance(text_prompt, (list, tuple)):
        return [str(item).strip() for item in text_prompt if str(item).strip()]

    if isinstance(text_prompt, str):
        return [item for item in (part.strip() for part in text_prompt.split(",")) if item]

    value = str(text_prompt).strip()

    return [value] if value else []


def to_numpy_f32(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu().numpy()

    return np.asarray(value, dtype=np.float32)


def to_numpy_u8(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def normalize_masks_np(masks: Any) -> np.ndarray:
    masks_np = to_numpy_u8(masks)

    if masks_np.ndim == 4:
        masks_np = masks_np[0]

    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    return (masks_np > 0).astype(np.uint8)


def resize_masks_to_hw(
    masks: Any,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    masks_nhw = normalize_masks_np(masks)

    n, h, w = masks_nhw.shape

    if h == target_h and w == target_w:
        return masks_nhw

    output = np.zeros((n, target_h, target_w), dtype=np.uint8)

    for idx in range(n):
        output[idx] = cv2.resize(
            masks_nhw[idx],
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return output


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def get_label_colors(labels: List[str]) -> Dict[str, Tuple[int, int, int]]:
    return {
        label: PALETTE[index % len(PALETTE)]
        for index, label in enumerate(labels)
    }


def build_label_color_legend(
    labels: List[str],
    label_colors: Dict[str, Tuple[int, int, int]],
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    legend = []

    for index, label in enumerate(labels):
        rgb = label_colors[label]

        instance_count = sum(
            1 for detection in detections if detection.get("label") == label
        )

        legend.append(
            {
                "label": label,
                "color_rgb": list(rgb),
                "color_hex": rgb_to_hex(rgb),
                "color_name": COLOR_NAMES[index % len(COLOR_NAMES)],
                "palette_index": index,
                "instance_count": instance_count,
            }
        )

    return legend


def mask_to_boxes(masks: Any) -> List[Tuple[int, int, int, int]]:
    masks_np = normalize_masks_np(masks)

    boxes = []

    for mask in masks_np:
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        boxes.append(
            (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            )
        )

    return boxes


def overlay_masks_by_label(
    image: Image.Image,
    masks: Any,
    mask_labels: List[str],
    label_colors: Dict[str, Tuple[int, int, int]],
    alpha: float = 0.5,
) -> Image.Image:
    image = image.convert("RGBA")

    if masks is None:
        return image.convert("RGB")

    masks_np = normalize_masks_np(masks)

    if masks_np.shape[0] == 0:
        return image.convert("RGB")

    n_masks = masks_np.shape[0]

    if len(mask_labels) != n_masks:
        mask_labels = (mask_labels + ["object"] * n_masks)[:n_masks]

    overlay_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for idx in range(n_masks):
        label = mask_labels[idx]
        color = label_colors.get(label, (255, 255, 0))

        mask_image = Image.fromarray((masks_np[idx] * 255).astype(np.uint8))

        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, resample=RESAMPLE_NEAREST)

        color_layer = Image.new("RGBA", image.size, color + (0,))
        mask_alpha = mask_image.point(lambda value, a=alpha: int(value * a) if value > 0 else 0)

        color_layer.putalpha(mask_alpha)

        overlay_layer = Image.alpha_composite(overlay_layer, color_layer)

    return Image.alpha_composite(image, overlay_layer).convert("RGB")


def draw_boxes(
    image: Image.Image,
    boxes_by_label: Dict[str, List[Tuple[int, int, int, int]]],
    label_colors: Dict[str, Tuple[int, int, int]],
    show_labels: bool = True,
    box_width: int = 3,
) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for label, boxes in boxes_by_label.items():
        color = label_colors.get(label, (255, 255, 0))
        multi = len(boxes) > 1

        for instance_index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box

            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

            if not show_labels:
                continue

            text = f"{label} #{instance_index}" if multi else str(label)
            pad = 2

            if font:
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_w = right - left
                text_h = bottom - top
            else:
                text_w = len(text) * 6
                text_h = 12

            tag_y1 = max(0, y1 - (text_h + 2 * pad))

            draw.rectangle(
                [x1, tag_y1, x1 + text_w + 2 * pad, tag_y1 + text_h + 2 * pad],
                fill=color,
            )

            draw.text(
                (x1 + pad, tag_y1 + pad),
                text,
                fill=(0, 0, 0),
                font=font,
            )

    return output


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def gaussian_blur_boxes_pil(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
) -> Image.Image:
    if not boxes:
        return image

    rgb = np.array(image.convert("RGB"))

    height, width = rgb.shape[:2]

    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = rgb[y1 : y2 + 1, x1 : x2 + 1]

        if roi.size == 0:
            continue

        base = max(roi.shape[0], roi.shape[1]) // 6
        kernel = max(15, min(99, base * 2 + 1))

        if kernel % 2 == 0:
            kernel += 1

        rgb[y1 : y2 + 1, x1 : x2 + 1] = cv2.GaussianBlur(
            roi,
            (kernel, kernel),
            0,
        )

    return Image.fromarray(rgb)


def apply_visuals_image(
    base_image: Image.Image,
    masks: Optional[np.ndarray],
    mask_labels: List[str],
    boxes_by_label: Dict[str, List[Tuple[int, int, int, int]]],
    label_colors: Dict[str, Tuple[int, int, int]],
    show_segmentation: bool,
    show_boxes: bool,
    show_labels: bool,
    blur_boxes: bool,
) -> Image.Image:
    output = base_image

    effective_show_segmentation = bool(show_segmentation) and not blur_boxes

    if blur_boxes:
        all_boxes = [box for boxes in boxes_by_label.values() for box in boxes]

        if all_boxes:
            output = gaussian_blur_boxes_pil(output, all_boxes)

        output = output.convert("RGB")

    else:
        if effective_show_segmentation and masks is not None:
            output = overlay_masks_by_label(
                output,
                masks,
                mask_labels,
                label_colors,
            )
        else:
            output = output.convert("RGB")

    if show_boxes and boxes_by_label:
        output = draw_boxes(
            output,
            boxes_by_label,
            label_colors,
            show_labels=bool(show_labels),
        )

    return output


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def format_detections_summary(result: Dict[str, Any]) -> str:
    detections = result.get("detections") or []

    lines = [
        f"Instances found: {result.get('object_count', len(detections))}",
        f"Labels: {', '.join(result.get('prompts_used') or [])}",
        f"Color strategy: {result.get('color_strategy', COLOR_STRATEGY)}",
        "",
        "Legend:",
    ]

    for entry in result.get("label_colors") or []:
        lines.append(
            f"  • {entry.get('label')} -> {entry.get('color_name')} "
            f"{entry.get('color_hex')} rgb={entry.get('color_rgb')} "
            f"— {entry.get('instance_count')} instance(s)"
        )

    lines.append("")
    lines.append("Instances:")

    for detection in detections:
        score = detection.get("score")
        score_text = f" score={float(score):.4f}" if score is not None else ""

        lines.append(
            f"  #{detection.get('id')} {detection.get('label')} "
            f"(instance {detection.get('instance_index')}) "
            f"color={detection.get('color_name')} {detection.get('color_hex')}"
            f"{score_text} bbox={detection.get('bbox')}"
        )

    return "\n".join(lines)


def segmentation_payload_for_export(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "status": result.get("status"),
        "color_strategy": result.get("color_strategy", COLOR_STRATEGY),
        "prompts_used": result.get("prompts_used") or [],
        "object_count": result.get("object_count", 0),
        "label_colors": result.get("label_colors") or [],
        "detections": result.get("detections") or [],
    }

    if result.get("status") != "success":
        payload["message"] = result.get("message")

    return payload


def segment_uploaded_image_text(
    image: UploadedFile,
    text_prompt: str,
    threshold: float = DEFAULT_THRESHOLD,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    show_segmentation: bool = True,
    show_boxes: bool = True,
    show_labels: bool = True,
    blur_boxes: bool = False,
) -> Dict[str, Any]:
    try:
        pil_image = load_uploaded_image(image)
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }

    prompts = normalize_text_prompts(text_prompt)

    if not prompts:
        return {
            "status": "error",
            "message": "Provide at least one text label. Example: person, car, dog",
        }

    try:
        model, processor = get_sam3_image_model()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load SAM3 model. Set HF_TOKEN for gated weights. Details: {e}",
        }

    orig_w, orig_h = pil_image.size

    mask_chunks: List[np.ndarray] = []
    mask_labels: List[str] = []
    detections: List[Dict[str, Any]] = []
    boxes_by_label: Dict[str, List[Tuple[int, int, int, int]]] = {}

    label_colors = get_label_colors(prompts)

    detection_id = 0

    for prompt_index, prompt in enumerate(prompts):
        try:
            inputs = processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(DEVICE) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with torch.no_grad(), autocast_context():
                outputs = model(**inputs)

            target_sizes = inputs.get("original_sizes")

            if target_sizes is None:
                target_sizes = [[orig_h, orig_w]]
            else:
                target_sizes = target_sizes.tolist()

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )[0]

            masks = results.get("masks")

            if masks is None:
                continue

            masks_nhw = resize_masks_to_hw(masks, orig_h, orig_w)
            boxes = mask_to_boxes(masks_nhw)

            boxes_by_label[prompt] = boxes

            scores_arr = (
                to_numpy_f32(results.get("scores"))
                if results.get("scores") is not None
                else np.array([])
            )

            rgb = label_colors[prompt]
            hex_color = rgb_to_hex(rgb)
            color_name = COLOR_NAMES[prompt_index % len(COLOR_NAMES)]

            for instance_index in range(masks_nhw.shape[0]):
                detection_id += 1

                score = (
                    float(scores_arr[instance_index])
                    if instance_index < len(scores_arr)
                    else None
                )

                bbox = (
                    list(boxes[instance_index])
                    if instance_index < len(boxes)
                    else None
                )

                detections.append(
                    {
                        "id": detection_id,
                        "label": prompt,
                        "instance_index": instance_index + 1,
                        "bbox": bbox,
                        "score": score,
                        "color_rgb": list(rgb),
                        "color_hex": hex_color,
                        "color_name": color_name,
                        "palette_index": prompt_index,
                    }
                )

                mask_labels.append(prompt)

            mask_chunks.append(masks_nhw)

        except Exception:
            continue

    if not mask_chunks:
        return {
            "status": "error",
            "message": "No objects found for the given prompt(s).",
            "object_count": 0,
            "prompts_used": prompts,
            "detections": [],
            "label_colors": [],
        }

    combined_masks = (
        np.concatenate(mask_chunks, axis=0)
        if len(mask_chunks) > 1
        else mask_chunks[0]
    )

    label_color_legend = build_label_color_legend(
        prompts,
        label_colors,
        detections,
    )

    result_image = apply_visuals_image(
        base_image=pil_image,
        masks=combined_masks,
        mask_labels=mask_labels,
        boxes_by_label=boxes_by_label,
        label_colors=label_colors,
        show_segmentation=show_segmentation,
        show_boxes=show_boxes,
        show_labels=show_labels,
        blur_boxes=blur_boxes,
    )

    result_image_bytes = pil_to_png_bytes(result_image)
    result_image_base64 = image_bytes_to_base64(result_image_bytes)

    object_count = len(detections)

    result = {
        "status": "success",
        "filename": getattr(image, "name", "image"),
        "content_type": "image/png",
        "width": result_image.width,
        "height": result_image.height,
        "color_strategy": COLOR_STRATEGY,
        "prompts_used": prompts,
        "object_count": object_count,
        "label_colors": label_color_legend,
        "detections": detections,
        "segmented_image_base64": result_image_base64,
        "device_info": device_info(),
    }

    result["detections_summary"] = format_detections_summary(result)

    return result


def build_segmentation_response(
    result: Dict[str, Any],
    include_base64: bool = True,
    include_summary: bool = True,
    include_device_info: bool = False,
) -> Dict[str, Any]:
    payload = segmentation_payload_for_export(result)

    if result.get("status") == "success":
        payload["filename"] = result.get("filename")
        payload["content_type"] = result.get("content_type")
        payload["width"] = result.get("width")
        payload["height"] = result.get("height")

    if include_base64:
        payload["segmented_image_base64"] = result.get("segmented_image_base64")

    if include_summary:
        payload["detections_summary"] = result.get("detections_summary")

    if include_device_info:
        payload["device_info"] = result.get("device_info")

    return payload