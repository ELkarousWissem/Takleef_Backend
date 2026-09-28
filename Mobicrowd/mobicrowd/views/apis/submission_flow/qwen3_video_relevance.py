
"""
OpenRouter Qwen3-VL-32B multi-frame caption + relevance with key rotation + logging.

Callable:
  openrouter_qwen3_vl_multiframe_caption_and_relevance(...)

Behavior:
- Accepts a folder path containing EXACTLY 8 frames
- Resizes each frame to 224x224 JPEG (small payload)
- Sends all 8 frames in ONE API call
- Returns clean dict:
  {
    ok, attempt, key_used, model,
    caption, relevant, recapture, reason,
    unwanted_content, unwanted_categories,
    latency_seconds, usage, cost,
    frames, openrouter_id
  }
"""

import os
import io
import time
import json
import random
from typing import List, Dict, Any, Tuple, Optional

import requests
from PIL import Image


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# -----------------------------
# Keys loader
# -----------------------------
def load_keys_from_file(path: str) -> List[str]:
    keys: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # support formats like: OPENROUTER_API_KEY=xxxx
            if "=" in s:
                s = s.split("=", 1)[1].strip()

            # support accidental comma-separated lines
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                keys.extend(parts)
            else:
                keys.append(s)

    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def mask_key(key: str) -> str:
    if not key:
        return "EMPTY"
    if len(key) < 12:
        return "***"
    return f"{key[:8]}...{key[-6:]}"


# -----------------------------
# Frames loading
# -----------------------------
def list_frames(frames_dir: str, expected: int = 8) -> List[str]:
    """
    Returns sorted frame paths (lexicographic sort) and enforces EXACTLY expected count.
    """
    if not os.path.isdir(frames_dir):
        raise NotADirectoryError(f"frames_dir is not a directory: {frames_dir}")

    files: List[str] = []
    for name in os.listdir(frames_dir):
        p = os.path.join(frames_dir, name)
        if os.path.isfile(p) and name.lower().endswith(ALLOWED_EXTS):
            files.append(p)

    files.sort()

    if len(files) != expected:
        raise ValueError(
            f"frames_dir must contain exactly {expected} image files. Found {len(files)}.\n"
            f"Directory: {frames_dir}\n"
            f"Files: {', '.join(os.path.basename(x) for x in files[:20])}"
        )

    return files


def get_file_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def image_to_data_url_224(
    image_path: str,
    size: int = 224,
    quality: int = 75,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resize shortest side to `size`, center-crop to `size x size`, JPEG encode.
    Returns (data_url, meta).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_bytes = get_file_size_bytes(image_path)

    img = Image.open(image_path).convert("RGB")
    ow, oh = img.size

    scale = size / float(min(ow, oh))
    nw, nh = int(ow * scale), int(oh * scale)
    img = img.resize((nw, nh), Image.LANCZOS)

    left = (nw - size) // 2
    top = (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    jpeg_bytes = buf.getbuffer().nbytes

    import base64
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    meta = {
        "path": image_path,
        "filename": os.path.basename(image_path),
        "original_width": ow,
        "original_height": oh,
        "original_file_bytes": orig_bytes,
        "resized_width": size,
        "resized_height": size,
        "resized_jpeg_bytes": jpeg_bytes,
        "jpeg_quality": quality,
        "data_url_chars": len(data_url),
    }
    return data_url, meta


def prepare_8_frames_data_urls(
    frames: List[str],
    size: int = 224,
    quality: int = 75,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    urls: List[str] = []
    metas: List[Dict[str, Any]] = []
    for p in frames:
        u, m = image_to_data_url_224(p, size=size, quality=quality)
        urls.append(u)
        metas.append(m)
    return urls, metas


UNWANTED_CATEGORIES = [
    "violence",
    "blood",
    "fights",
    "aggressive_acts",
    "guns_weapons",
    "visible_injuries",
    "nudity",
    "sexual_content",
    "political_content",
    "slogans",
    "posters",
    "flags",
    "graffiti",
    "brands_logos",
    "company_names",
    "recognizable_trademarks",
    "screen_recapture",
]

UNWANTED_CATEGORY_DESCRIPTIONS = {
    "violence": "violent scenes, riots, war aftermath, assaults",
    "blood": "blood, crime scenes, bloody wounds",
    "fights": "street fights, brawls, boxing, MMA",
    "aggressive_acts": "aggressive confrontations, road rage, angry mobs",
    "guns_weapons": "guns, pistols, rifles, knives, military weapons",
    "visible_injuries": "bruises, wounds, stitches, black eyes, bandaged injuries",
    "nudity": "visible private body parts, topless or nude person",
    "sexual_content": "sexual acts, erotic poses, intimate touching,intimate scenes,intimate positions, intimate interactions,intimate movements, intimate expressions",
    "political_content": "political rallies, election events, political protests",
    "slogans": "protest slogans, political banners, activist placards",
    "posters": "political campaign posters, election posters, propaganda posters",
    "flags": "political flags at protests, demonstrations with flags",
    "graffiti": "political graffiti, protest messages sprayed on walls",
    "brands_logos": "brand logos, company logos on storefronts or products",
    "company_names": "company name signs on buildings, corporate signage",
    "recognizable_trademarks": "trademark logos on packaging, famous brand marks",
    "screen_recapture": (
        "the submitter cheated by photographing a reproduction instead of capturing the "
        "requested real-world subject — e.g. a phone/tablet/monitor showing an image, a "
        "printed photo held up to the camera, or moiré/pixel-grid/bezel-dominated capture "
        "where the image is clearly a photo-of-a-photo or photo-of-a-screen rather than a "
        "direct live capture. NOT recapture when the user explicitly requested screens, "
        "monitors, TVs, phones, tablets, or displays as the subject and the photo shows "
        "those physical devices in a real environment"
    ),
}


# -----------------------------
# Prompt (accurate + constraints + anti-hallucination)
# -----------------------------
def build_prompt(description: str) -> str:
    unwanted_block = "\n".join(
        f"  - {cat}: {UNWANTED_CATEGORY_DESCRIPTIONS[cat]}"
        for cat in UNWANTED_CATEGORIES
    )
    return f"""
Role: You are an expert Vision Captioner and Intuitive Relevance Judge. You analyze visual data with the nuance of a human, understanding that "related" items often fulfill a user's intent even if specific keywords differ.

Input: 8 frames from the same scene + a User Description.

Output: VALID JSON ONLY (no extra text).

0) GENERAL SAFETY/CONTENT GATE (must run first)
Review ALL 8 frames together. Detect whether the scene contains ANY unwanted categories below
in any frame:
{unwanted_block}

If any unwanted category is detected in any frame:
- set "unwanted_content" = "YES"
- list exact category names in "unwanted_categories"
- set "relevant" = "NO" (always reject by general rules)

1) RECAPTURE CHECK — MUST ALWAYS BE DECIDED
Review ALL 8 frames together and determine whether the submitted video is a reproduction
rather than a direct recording of the requested real-world subject.

Set "recapture" = true when the frames clearly show:
- another screen, monitor, TV, phone, tablet, or laptop displaying the requested content;
- a printed photograph or printed image being used instead of the real subject;
- a video/photo-of-a-photo, screen recording reproduction, or another reproduced visual;
- strong reproduction evidence such as a screen bezel, UI, pixel grid, moiré, or a
  displayed image/video instead of the real subject.

Set "recapture" = false for a genuine direct real-world recording.

IMPORTANT exception: if the USER DESCRIPTION explicitly asks for screens, monitors, TVs,
phones, tablets, laptops, kiosks, digital signage, or displays as physical subjects, a
genuine recording of those physical devices in the real environment is NOT a recapture.

The "recapture" field MUST always be present and MUST be a JSON boolean true or false.
Never return "YES"/"NO" or string "true"/"false" for this field.

If "recapture" = true:
- include "screen_recapture" in "unwanted_categories";
- set "unwanted_content" = "YES";
- set "relevant" = "NO".

Examples:
- Task: "Capture video of TV displays in an electronics store" + frames show real TVs on
  a shop wall → recapture=false, unwanted_content=NO, relevant=YES
- Task: "Record cats playing" + frames show a cat video on a phone screen → recapture=true,
  unwanted_content=YES, unwanted_categories=["screen_recapture"], relevant=NO
- Task: "Record computer monitors in an office" + frames show monitors on desks in a
  live office → recapture=false, unwanted_content=NO, relevant=YES
- Task: "Record an office workspace" + frames are dominated by a laptop screen displaying
  an office video → recapture=true, unwanted_content=YES,
  unwanted_categories=["screen_recapture"], relevant=NO

2) CAPTION (Anti-Hallucination)
Granular Detail: Describe ONLY what is clearly visible across the 8 frames. Do not guess what is behind or inside objects.

Explicit Naming: Name every distinct object. Avoid vague terms like "various items" or "clutter."

Strict Counting: Include counts (e.g., "3 chairs") only if they are clearly countable in the frames. Otherwise, omit the number.

Text Extraction: Transcribe any visible text exactly as written (e.g., logos, signs, labels).

3) RELEVANCE (Contextual Intelligence)
Thematic Matching: Use human-like reasoning to determine if the scene matches the intent of the description. If the user asks for a "workspace," classify a "kitchen table with a laptop" as Related, even if the word "office" wasn't used.

Example Logic: Treat examples in the description as representative, not exhaustive. If a user lists "dogs, cats," recognize that a "hamster" is a related pet contextually, unless specifically excluded.

Strict Constraint Enforcement: Rigorously apply negations. If the description says "no people" or "reject if outdoors," any presence of these elements results in a Not Relevant status, regardless of other matching items.

Reasoning: Briefly explain why the scene is related or not, noting any synonym matches or constraint violations.

USER DESCRIPTION:
{description}

RETURN JSON:
{{
  "caption": "1-2 short factual sentences.",
  "recapture": false,
  "unwanted_content": "YES|NO",
  "unwanted_categories": ["exact_category_name"],
  "relevant": "YES|NO",
  "reason": "max 20 words: key evidence or violated constraint"
}}

"recapture" MUST be a JSON boolean: true or false. Never quote it.
""".strip()


def parse_model_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw_text or not raw_text.strip():
        return None, "Empty model output."

    s = raw_text.strip()

    # direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "Model JSON is not an object."
    except json.JSONDecodeError:
        pass

    # best-effort extraction
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
            return None, "Extracted JSON is not an object."
        except json.JSONDecodeError:
            return None, "Failed to parse extracted JSON."

    return None, "No JSON object found."


def norm_yesno(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s.startswith("Y"):
        return "YES"
    if s.startswith("N"):
        return "NO"
    return "NO"


def norm_bool(value: Any) -> bool:
    """Normalize model output to a real Python bool for the recapture contract."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "y"}:
            return True
        if normalized in {"false", "no", "0", "n", ""}:
            return False
    return False


def normalise_categories(value: Any) -> List[str]:
    if value is None:
        values: List[Any] = []
    elif isinstance(value, str):
        values = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    allowed = {category.lower(): category for category in UNWANTED_CATEGORIES}
    categories: List[str] = []
    for item in values:
        key = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        category = allowed.get(key)
        if category and category not in categories:
            categories.append(category)
    return categories


# -----------------------------
# OpenRouter request (one try) with 8 frames
# -----------------------------
def openrouter_call_once_multiframe(
    api_key: str,
    frame_data_urls: List[str],
    description: str,
    model_id: str,
    fallback_models: Optional[List[str]] = None,
    max_tokens: int = 220,
    timeout_s: int = 180,
) -> Tuple[bool, Dict[str, Any]]:
    prompt = build_prompt(description)

    content = [{"type": "text", "text": prompt}]
    for url in frame_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "provider": {"allow_fallbacks": True},
    }
    if fallback_models:
        payload["models"] = list(fallback_models)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Qwen3-VL-32B MultiFrame Caption+Relevance",
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(
            OPENROUTER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
    except requests.Timeout:
        return False, {"type": "timeout", "message": "Request timed out."}
    except requests.RequestException as e:
        return False, {"type": "request_exception", "message": str(e)}

    latency = time.perf_counter() - t0

    try:
        data = r.json()
    except Exception:
        return False, {
            "type": "bad_json",
            "http_status": r.status_code,
            "latency_seconds": round(latency, 4),
            "raw_text": (r.text or "")[:2000],
        }

    # non-2xx
    if not (200 <= r.status_code < 300):
        retry_after = data.get("error", {}).get("metadata", {}).get("retry_after_seconds")
        return False, {
            "type": "http_error",
            "http_status": r.status_code,
            "latency_seconds": round(latency, 4),
            "retry_after_seconds": retry_after,
            "openrouter_response": data,
        }

    # provider error inside 200
    if "error" in data and "choices" not in data:
        retry_after = data.get("error", {}).get("metadata", {}).get("retry_after_seconds")
        return False, {
            "type": "provider_error",
            "latency_seconds": round(latency, 4),
            "retry_after_seconds": retry_after,
            "openrouter_response": data,
        }

    if "choices" not in data:
        return False, {
            "type": "missing_choices",
            "latency_seconds": round(latency, 4),
            "openrouter_response": data,
        }

    raw_text = data["choices"][0]["message"].get("content", "") or ""
    parsed, err = parse_model_json(raw_text)

    usage = data.get("usage", {}) or {}
    usage_out = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
        "cost_details": usage.get("cost_details"),
    }

    if parsed is None:
        return False, {
            "type": "bad_model_output",
            "latency_seconds": round(latency, 4),
            "parse_error": err,
            "raw_model_text": raw_text[:2000],
            "usage": usage_out,
        }

    caption = str(parsed.get("caption", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    recapture = norm_bool(parsed.get("recapture"))
    unwanted_categories = normalise_categories(parsed.get("unwanted_categories"))
    unwanted_content = norm_yesno(parsed.get("unwanted_content"))

    # Canonical consistency rules:
    # - `recapture` is the dedicated downstream contract.
    # - keep the legacy screen_recapture category synchronized for compatibility.
    if recapture:
        if "screen_recapture" not in unwanted_categories:
            unwanted_categories.append("screen_recapture")
        unwanted_content = "YES"

    if "screen_recapture" in unwanted_categories:
        recapture = True
        unwanted_content = "YES"

    if unwanted_categories:
        unwanted_content = "YES"

    relevant = (
        "NO"
        if recapture or unwanted_content == "YES"
        else norm_yesno(parsed.get("relevant"))
    )

    return True, {
        "caption": caption,
        "relevant": relevant,
        "recapture": bool(recapture),
        "unwanted_content": unwanted_content,
        "unwanted_categories": unwanted_categories,
        "reason": reason,
        "latency_seconds": round(latency, 4),
        "usage": usage_out,
        "cost": usage_out.get("cost"),
        "openrouter_id": data.get("id"),
        "model": data.get("model") or model_id,
        "raw_model_text": raw_text,
    }


def should_rotate(err: Dict[str, Any]) -> bool:
    t = err.get("type", "")

    if t in ("timeout", "request_exception", "provider_error", "missing_choices", "bad_model_output"):
        return True

    if t == "http_error":
        status = err.get("http_status")
        # Configuration/auth/request errors are not repaired by another key.
        # OpenRouter already exhausted the configured model fallback chain.
        if status in (400, 401, 403, 404, 413, 422):
            return False
        if status == 429:
            return True
        if status and status >= 500:
            return True
        return False

    return True


def backoff_sleep(attempt: int, retry_after_seconds: Optional[float], base_backoff: float) -> None:
    if retry_after_seconds is not None:
        time.sleep(float(retry_after_seconds) + random.random() * 0.1)
        return
    time.sleep(base_backoff * (2 ** (attempt - 1)) + random.random() * 0.2)


# ==========================================================
# ✅ Callable function (IMPORT THIS IN CELERY)
# ==========================================================
def openrouter_qwen3_vl_multiframe_caption_and_relevance(
    *,
    frames_dir: str,
    description: str,
    keys_file: str,
    logger,
    model_id: str,
    fallback_models: Optional[List[str]] = None,
    expected_frames: int = 8,
    max_attempts: int = 10,
    max_tokens: int = 220,
    jpeg_quality: int = 75,
    base_backoff: float = 1.4,
    timeout_s: int = 180,
) -> Dict[str, Any]:
    """
    Returns dict:

    Success:
      {
        "ok": True,
        "attempt": int,
        "key_used": "masked",
        "model": str,
        "caption": str,
        "relevant": "YES|NO",
        "recapture": bool,
        "unwanted_content": "YES|NO",
        "unwanted_categories": ["category_name"],
        "reason": str,
        "latency_seconds": float,
        "usage": {...},
        "cost": float|None,
        "frames": [meta...],
        "openrouter_id": str|None,
      }

    Failure:
      {
        "ok": False,
        "recapture": False,
        "reason": "...",
        "attempts": int,
        "last_error": {...},
        "frames": [meta... maybe empty]
      }
    """
    t0 = time.perf_counter()

    if not frames_dir or not os.path.isdir(frames_dir):
        return {
            "ok": False,
            "recapture": False,
            "reason": f"frames_dir not found or not a directory: {frames_dir}",
            "attempts": 0,
            "last_error": {"type": "bad_input"},
            "frames": [],
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }

    if not keys_file or not os.path.exists(keys_file):
        return {
            "ok": False,
            "recapture": False,
            "reason": f"keys_file not found: {keys_file}",
            "attempts": 0,
            "last_error": {"type": "bad_input"},
            "frames": [],
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }

    keys = load_keys_from_file(keys_file)
    if not keys:
        return {
            "ok": False,
            "recapture": False,
            "reason": "No keys found in keys_file.",
            "attempts": 0,
            "last_error": {"type": "no_keys"},
            "frames": [],
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }

    # Prepare frames once
    try:
        frame_paths = list_frames(frames_dir, expected=expected_frames)
        frame_data_urls, frame_metas = prepare_8_frames_data_urls(
            frame_paths,
            size=224,
            quality=max(25, min(95, int(jpeg_quality))),
        )
    except Exception as e:
        return {
            "ok": False,
            "recapture": False,
            "reason": f"Failed to load/prepare frames: {e}",
            "attempts": 0,
            "last_error": {"type": "frames_prepare_error", "message": str(e)},
            "frames": [],
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }

    last_error: Optional[Dict[str, Any]] = None
    key_index = 0

    for attempt in range(1, max_attempts + 1):
        key = keys[key_index % len(keys)]
        key_index += 1

        logger.info(
            "[openrouter_multiframe] attempt=%s/%s key=%s model=%s frames=%s",
            attempt,
            max_attempts,
            mask_key(key),
            model_id,
            len(frame_data_urls),
        )

        ok, resp = openrouter_call_once_multiframe(
            api_key=key,
            frame_data_urls=frame_data_urls,
            description=(description or "").strip(),
            model_id=model_id,
            fallback_models=fallback_models,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

        if ok:
            out = {
                "ok": True,
                "attempt": attempt,
                "key_used": mask_key(key),
                "model": resp.get("model", model_id),
                "caption": resp.get("caption", ""),
                "relevant": resp.get("relevant", "NO"),
                "recapture": bool(resp.get("recapture", False)),
                "unwanted_content": resp.get("unwanted_content", "NO"),
                "unwanted_categories": resp.get("unwanted_categories", []),
                "reason": resp.get("reason", ""),
                "latency_seconds": resp.get("latency_seconds"),
                "usage": resp.get("usage", {}),
                "cost": resp.get("cost"),
                "frames": frame_metas,
                "openrouter_id": resp.get("openrouter_id"),
            }

            # concise success log
            try:
                usage = out.get("usage") or {}
                logger.info(
                    "[openrouter_multiframe] classification relevant=%s recapture=%s "
                    "unwanted=%s categories=%s reason=%r",
                    out["relevant"],
                    out["recapture"],
                    out["unwanted_content"],
                    out["unwanted_categories"],
                    out["reason"],
                )
                logger.info(
                    "[openrouter_multiframe] success attempt=%s key=%s relevant=%s "
                    "recapture=%s unwanted=%s tokens=%s cost=%s latency=%.3fs",
                    attempt,
                    out["key_used"],
                    out["relevant"],
                    out["recapture"],
                    out["unwanted_content"],
                    usage.get("total_tokens"),
                    out.get("cost"),
                    float(out.get("latency_seconds") or 0.0),
                )
            except Exception:
                pass

            return out

        # failure
        last_error = resp
        rotate = should_rotate(resp)
        retry_after = resp.get("retry_after_seconds")

        logger.warning(
            "[openrouter_multiframe] fail attempt=%s key=%s type=%s rotate=%s retry_after=%s",
            attempt,
            mask_key(key),
            resp.get("type"),
            rotate,
            retry_after,
        )

        if not rotate:
            return {
                "ok": False,
                "recapture": False,
                "reason": "Non-rotatable failure (payload/format).",
                "attempts": attempt,
                "last_error": last_error,
                "frames": frame_metas,
                "elapsed_s": round(time.perf_counter() - t0, 4),
            }

        backoff_sleep(attempt, retry_after, base_backoff)

    return {
        "ok": False,
        "recapture": False,
        "reason": "All attempts failed (key rotation exhausted).",
        "attempts": max_attempts,
        "last_error": last_error,
        "frames": frame_metas,
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }