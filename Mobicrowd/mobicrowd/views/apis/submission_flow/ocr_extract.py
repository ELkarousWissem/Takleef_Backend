from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from openai import OpenAI
from rest_framework import status, serializers
from rest_framework.response import Response
from rest_framework.views import APIView
from sympy.physics.units import frequency

from mobicrowd.models.submisson import Photo

logger = logging.getLogger(__name__)

OCR_SYSTEM_PROMPT = (
    "You are an OCR specialist. Extract ALL visible text from the image exactly as it appears. "
    "Preserve the original layout, line breaks, and formatting as closely as possible. "
    "If no text is found, reply with: [No text detected]"
)


def _load_keys_from_file(keys_file: str | os.PathLike) -> List[str]:
    path = Path(keys_file)
    if not path.exists():
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {path}")

    keys: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line)

    if not keys:
        raise RuntimeError(f"No valid OpenRouter keys found in: {path}")

    return keys


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def _exception_http_status(exc: Exception):
    """Best-effort HTTP status extraction from OpenAI-compatible SDK errors."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def _uploaded_image_to_data_url(uploaded_file) -> str:
    content_type = getattr(uploaded_file, "content_type", None) or "image/jpeg"
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def _extract_message_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(content)

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def sanitize_text_for_legacy_mysql(text: str) -> str:
    """
    Remove characters that legacy MySQL utf8 cannot store
    (mainly 4-byte Unicode such as emoji), while keeping normal text,
    Arabic, accents, punctuation, and line breaks.
    """
    if not text:
        return ""

    cleaned = []
    previous_was_removed = False

    for ch in text:
        code = ord(ch)

        # Remove NUL byte
        if code == 0:
            previous_was_removed = True
            continue

        # Keep BMP only (U+0000 to U+FFFF)
        if code <= 0xFFFF:
            cleaned.append(ch)
            previous_was_removed = False
        else:
            # Replace removed non-BMP chars with a space only if needed
            if cleaned and not cleaned[-1].isspace() and not previous_was_removed:
                cleaned.append(" ")
            previous_was_removed = True

    text = "".join(cleaned)

    # Clean spacing introduced by removed emoji/symbols
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def openrouter_ocr_with_rotation(
    *,
    uploaded_file,
    keys_file: str | os.PathLike,
    model_id: str,
    fallback_models: List[str] | None = None,
    max_attempts: int = 6,
    max_tokens: int = 2048,
    logger=None,
) -> Dict[str, Any]:
    keys = _load_keys_from_file(keys_file)
    data_url = _uploaded_image_to_data_url(uploaded_file)

    last_error = None

    for attempt in range(1, max_attempts + 1):
        api_key = keys[(attempt - 1) % len(keys)]
        t0 = time.perf_counter()

        try:
            client = OpenAI(
                base_url=getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=api_key,
            )

            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": "Extract all text from this image."},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                frequency_penalty=0.5,
                extra_body={"models": list(fallback_models)} if fallback_models else None,
            )

            latency = float(time.perf_counter() - t0)
            text = _extract_message_text(
                response.choices[0].message.content if response.choices else ""
            )

            if logger:
                logger.info(
                    "[openrouter_ocr_with_rotation] success attempt=%s model=%s key=%s latency_s=%.3f",
                    attempt,
                    model_id,
                    _mask_key(api_key),
                    latency,
                )

            actual_model = getattr(response, "model", None) or model_id
            return {
                "ok": True,
                "text": text or "[No text detected]",
                "model": actual_model,
                "requested_model": model_id,
                "configured_fallback_models": list(fallback_models or []),
                "model_fallback_used": bool(actual_model and actual_model != model_id),
                "attempt": attempt,
                "key_used": _mask_key(api_key),
                "latency_seconds": latency,
                "usage": getattr(response, "usage", None),
                "openrouter_id": getattr(response, "id", None),
                "error": None,
            }

        except Exception as exc:
            latency = float(time.perf_counter() - t0)
            last_error = str(exc)
            http_status = _exception_http_status(exc)

            if logger:
                logger.warning(
                    "[openrouter_ocr_with_rotation] failed attempt=%s model=%s key=%s "
                    "http=%s latency_s=%.3f err=%s",
                    attempt,
                    model_id,
                    _mask_key(api_key),
                    http_status,
                    latency,
                    last_error,
                )

            # OpenRouter has already tried the configured model fallback chain.
            # These errors are not repaired by repeating the identical request.
            if http_status in (400, 401, 403, 404, 413, 422):
                return {
                    "ok": False,
                    "text": "",
                    "model": model_id,
                    "attempt": attempt,
                    "key_used": _mask_key(api_key),
                    "latency_seconds": latency,
                    "usage": None,
                    "openrouter_id": None,
                    "error": last_error,
                    "reason": f"http_{http_status}",
                }

    return {
        "ok": False,
        "text": "",
        "model": model_id,
        "attempt": max_attempts,
        "key_used": None,
        "latency_seconds": None,
        "usage": None,
        "openrouter_id": None,
        "error": last_error,
        "reason": "all_attempts_failed",
    }


def extract_ocr_from_uploaded_image(uploaded_file, logger=None) -> Dict[str, Any]:
    keys_file = settings.OPENROUTER_KEYS_FILE
    model_id = settings.OPENROUTER_OCR_MODEL
    fallback_models = list(
        getattr(settings, "OPENROUTER_OCR_FALLBACK_MODELS", [])
    )
    max_attempts = int(getattr(settings, "OPENROUTER_OCR_MAX_ATTEMPTS", 6))
    max_tokens = int(getattr(settings, "OPENROUTER_OCR_MAX_TOKENS", 512))

    if not keys_file:
        raise RuntimeError("OPENROUTER_KEYS_FILE is not configured.")
    if not os.path.exists(keys_file):
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {keys_file}")
    if not model_id:
        raise RuntimeError("OPENROUTER_OCR_MODEL is not configured.")

    return openrouter_ocr_with_rotation(
        uploaded_file=uploaded_file,
        keys_file=keys_file,
        model_id=model_id,
        fallback_models=fallback_models,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
        logger=logger,
    )


class PhotoOCRBatchSerializer(serializers.Serializer):
    photo_ids = serializers.ListField(
        child=serializers.IntegerField(),
        allow_empty=False,
        required=True,
    )


class ExtractTextApprovedPhotosAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = PhotoOCRBatchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        photo_ids = serializer.validated_data["photo_ids"]

        photos = Photo.objects.filter(id__in=photo_ids)
        photo_map = {int(p.id): p for p in photos}

        results = []

        for photo_id in photo_ids:
            photo = photo_map.get(int(photo_id))
            if not photo or not getattr(photo, "image", None):
                results.append({
                    "photo_id": photo_id,
                    "ok": False,
                    "error": "photo not found or missing image",
                })
                continue

            try:
                photo.image.open("rb")
                content = photo.image.read()
                photo.image.close()

                content_type = mimetypes.guess_type(photo.image.name)[0] or "image/jpeg"

                uploaded = SimpleUploadedFile(
                    name=os.path.basename(photo.image.name),
                    content=content,
                    content_type=content_type,
                )

                res = extract_ocr_from_uploaded_image(uploaded, logger=logger)

                if not res.get("ok"):
                    results.append({
                        "photo_id": photo_id,
                        "ok": False,
                        "error": res.get("error") or "ocr_failed",
                        "reason": res.get("reason"),
                        "analysis": {
                            "model": res.get("model"),
                            "attempt": res.get("attempt"),
                            "key_used": res.get("key_used"),
                            "latency_seconds": res.get("latency_seconds"),
                        },
                    })
                    continue

                raw_extracted_text = (res.get("text") or "").strip() or "[No text detected]"
                extracted_text = sanitize_text_for_legacy_mysql(raw_extracted_text)

                if not extracted_text:
                    extracted_text = "[No text detected]"

                logger.info(
                    "OCR sanitize photo_id=%s raw_len=%s cleaned_len=%s",
                    photo_id,
                    len(raw_extracted_text),
                    len(extracted_text),
                )

                photo.extracted_text = extracted_text
                photo.save(update_fields=["extracted_text"])

                results.append({
                    "photo_id": photo_id,
                    "ok": True,
                    "text": extracted_text,
                    "analysis": {
                        "model": res.get("model"),
                        "attempt": res.get("attempt"),
                        "key_used": res.get("key_used"),
                        "latency_seconds": res.get("latency_seconds"),
                        "usage": res.get("usage"),
                        "openrouter_id": res.get("openrouter_id"),
                    },
                })

            except Exception as exc:
                logger.exception("OCR batch failed for photo_id=%s", photo_id)
                results.append({
                    "photo_id": photo_id,
                    "ok": False,
                    "error": str(exc),
                })

        return Response(
            {
                "ok": True,
                "results": results,
            },
            status=status.HTTP_200_OK,
        )