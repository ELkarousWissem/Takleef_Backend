import base64
import os
from io import BytesIO
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np
import torch
from django.core.files.uploadedfile import UploadedFile
from PIL import Image, ImageOps
from transformers import AutoModelForImageSegmentation


MODEL_ID = os.environ.get("BACKGROUND_REMOVAL_MODEL_ID", "ZhengPeng7/BiRefNet")

DEVICE = torch.device(
    "cuda"
    if os.environ.get("BACKGROUND_REMOVAL_DEVICE", "").lower() == "cuda"
    and torch.cuda.is_available()
    else "cpu"
)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

_MODEL: Optional[torch.nn.Module] = None
_MODEL_LOCK = Lock()

MAX_IMAGE_SIZE_MB = 10

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def set_device(device: str) -> None:
    global DEVICE, _MODEL

    selected_device = torch.device(device)

    if selected_device.type == "cuda" and not torch.cuda.is_available():
        selected_device = torch.device("cpu")

    if DEVICE != selected_device:
        _MODEL = None

    DEVICE = selected_device


def get_model() -> torch.nn.Module:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None:
            model = AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )

            # Force model dtype to float32 to match input tensor dtype
            model = model.to(device=DEVICE, dtype=torch.float32)
            model.eval()

            _MODEL = model

    return _MODEL


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


def prepare_image(image: Image.Image) -> torch.Tensor:
    resized = image.resize((1024, 1024), RESAMPLE_BILINEAR)

    image_array = np.asarray(resized, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

    mean = MEAN.to(dtype=image_tensor.dtype)
    std = STD.to(dtype=image_tensor.dtype)

    image_tensor = (image_tensor - mean) / std

    return image_tensor.unsqueeze(0).to(device=DEVICE, dtype=torch.float32)


def extract_prediction_tensor(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, (list, tuple)):
        return model_output[-1]

    if hasattr(model_output, "logits"):
        return model_output.logits

    raise ValueError("Unsupported model output format.")


def remove_background_from_pil(image: Image.Image) -> Image.Image:
    model = get_model()

    original_size = image.size
    model_input = prepare_image(image)

    with torch.inference_mode():
        model_output = model(model_input)
        prediction = extract_prediction_tensor(model_output)
        prediction = prediction.sigmoid().detach().cpu()[0].squeeze()

    mask_array = (prediction.numpy() * 255).clip(0, 255).astype(np.uint8)

    mask = Image.fromarray(mask_array).convert("L")
    mask = mask.resize(original_size, RESAMPLE_BILINEAR)

    output = image.copy().convert("RGBA")
    output.putalpha(mask)

    return output


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def run_background_removal(
    image: UploadedFile,
    use_cuda: bool = False,
) -> Dict[str, Any]:
    if use_cuda:
        set_device("cuda")
    else:
        set_device("cpu")

    original_image = load_uploaded_image(image)
    transparent_image = remove_background_from_pil(original_image)

    png_bytes = pil_to_png_bytes(transparent_image)
    png_base64 = png_bytes_to_base64(png_bytes)

    return {
        "status": "success",
        "filename": getattr(image, "name", "image"),
        "content_type": "image/png",
        "width": transparent_image.width,
        "height": transparent_image.height,
        "removed_background_base64": png_base64,
    }