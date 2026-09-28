"""
OpenRouter Qwen3-VL caption + relevance with:
- API key rotation from file
- retries + backoff
- image resizing (224x224 center crop)
- logs: latency, tokens, cost, attempt, key used (masked), image bytes

Return structure (clean):
{
  "ok": bool,
  "attempt": int,
  "key_used": str (masked),
  "model": str,
  "caption": str,
  "relevant": "YES|NO",
  "recapture": bool,
  "reason": str,
  "latency_seconds": float,
  "usage": {...},
  "cost": float | None,
  "image": {...},
  "openrouter_id": str | None,
  "error": {...}  # only if ok=False
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


# -----------------------------
# Keys
# -----------------------------
def load_keys_from_file(path: str) -> List[str]:
    keys: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # allow KEY=...
            if "=" in s:
                s = s.split("=", 1)[1].strip()
            # allow comma-separated
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                keys.extend(parts)
            else:
                keys.append(s)

    # de-dup preserve order
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
# Image utilities
# -----------------------------
def _file_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def image_to_data_url_224(
    image_path: str,
    size: int = 224,
    quality: int = 75,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resize shortest side -> size, center-crop size x size, JPEG encode -> data_url.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_bytes = _file_size_bytes(image_path)

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
# Prompt
# -----------------------------
def build_prompt(description: str) -> str:
    unwanted_block = "\n".join(
        f"  - {cat}: {UNWANTED_CATEGORY_DESCRIPTIONS[cat]}"
        for cat in UNWANTED_CATEGORIES
    )
    return f"""
Role: You are an expert Vision Captioner and Intuitive Relevance Judge. You analyze visual data with the nuance of a human, understanding that "related" items often fulfill a user's intent even if specific keywords differ.

Input: Images from a single scene + a User Description.

Output: VALID JSON ONLY.

0) GENERAL SAFETY/CONTENT GATE (must run first)
Detect whether the scene contains ANY unwanted categories below:
{unwanted_block}

If any unwanted category is detected:
- set "unwanted_content" = "YES"
- list exact category names in "unwanted_categories"
- set "relevant" = "NO" (always reject by general rules)

1) RECAPTURE CHECK — MUST ALWAYS BE DECIDED
Determine whether the submitted image is a reproduction rather than a direct capture of
the requested real-world subject.

Set "recapture" = true when the submission is clearly:
- a photo of another screen, monitor, TV, phone, tablet, or laptop displaying the requested content;
- a photo of a printed photograph or printed image;
- a photo-of-a-photo or another reproduced visual;
- dominated by reproduction evidence such as a screen bezel, UI, pixel grid, moiré, or a
  displayed image instead of the real subject.

Set "recapture" = false for a genuine direct real-world capture.

IMPORTANT exception: if the USER DESCRIPTION explicitly asks for screens, monitors, TVs,
phones, tablets, laptops, kiosks, digital signage, or displays as physical subjects, a
genuine photograph of those physical devices in the real environment is NOT a recapture.

The "recapture" field MUST always be present and MUST be a JSON boolean true or false.
Never return "YES"/"NO" or string "true"/"false" for this field.

If "recapture" = true:
- include "screen_recapture" in "unwanted_categories";
- set "unwanted_content" = "YES";
- set "relevant" = "NO".

Examples:
- Task: "Capture photos of TV displays in an electronics store" + image shows real TVs on
  a shop wall → recapture=false, unwanted_content=NO, relevant=YES
- Task: "Capture photos of cats" + image shows a cat on a phone screen → recapture=true,
  unwanted_content=YES, unwanted_categories=["screen_recapture"], relevant=NO
- Task: "Photograph computer monitors in an office" + image shows monitors on desks in a
  live office → recapture=false, unwanted_content=NO, relevant=YES
- Task: "Photograph computer monitors" + image is a photo of a laptop screen showing a
  monitor wallpaper → recapture=true, unwanted_content=YES,
  unwanted_categories=["screen_recapture"], relevant=NO

2) CAPTION (Anti-Hallucination)
Granular Detail: Describe only what is definitively visible. Do not guess what is behind or inside objects.

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

IMPORTANT: "recapture" MUST be a JSON boolean true or false, never a string.
""".strip()

def _parse_model_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw_text or not raw_text.strip():
        return None, "Empty model output."

    s = raw_text.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "Model JSON is not an object."
    except json.JSONDecodeError:
        pass

    # extract best-effort
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


def _norm_yesno(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s.startswith("Y"):
        return "YES"
    if s.startswith("N"):
        return "NO"
    return "NO"


def _norm_bool(value: Any) -> bool:
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


def _normalise_categories(value: Any) -> List[str]:
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
# Single OpenRouter call
# -----------------------------
def _openrouter_call_once(
    *,
    session: requests.Session,
    api_key: str,
    model_id: str,
    fallback_models: Optional[List[str]],
    image_data_url: str,
    description: str,
    max_tokens: int = 220,
    timeout_s: float = 25.0,  # ✅ faster
) -> Tuple[bool, Dict[str, Any]]:

    payload = {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(description)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "provider": {"allow_fallbacks": True},
    }
    if fallback_models:
        payload["models"] = list(fallback_models)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Mobicrowd VLM Relevance",
    }

    t0 = time.perf_counter()
    try:
        # ✅ split connect+read timeout
        r = session.post(
            OPENROUTER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=(5.0, float(timeout_s)),  # connect=5s, read=timeout_s
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
    parsed, err = _parse_model_json(raw_text)

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

    recapture = _norm_bool(parsed.get("recapture"))
    unwanted_categories = _normalise_categories(parsed.get("unwanted_categories"))
    unwanted_content = _norm_yesno(parsed.get("unwanted_content"))

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
        else _norm_yesno(parsed.get("relevant"))
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
    }


def _should_rotate(err: Dict[str, Any]) -> bool:
    t = err.get("type", "")

    if t in ("timeout", "request_exception", "provider_error", "missing_choices", "bad_model_output"):
        return True

    if t == "http_error":
        status = err.get("http_status")
        if status in (401, 403, 429):
            return True
        if status and status >= 500:
            return True

    # payload too large won't be fixed by rotating keys
    resp = err.get("openrouter_response", {}) or {}
    raw = ""
    try:
        raw = resp.get("error", {}).get("metadata", {}).get("raw", "")
    except Exception:
        raw = ""
    if "Request entity too large" in raw:
        return False

    return True


def _backoff_sleep(attempt: int, retry_after_seconds: Optional[float], base_backoff: float) -> None:
    if retry_after_seconds is not None:
        time.sleep(float(retry_after_seconds) + random.random() * 0.1)
        return
    time.sleep(base_backoff * (2 ** (attempt - 1)) + random.random() * 0.2)


# ==========================================================
# ✅ Public callable function (Celery-friendly)
# ==========================================================
def openrouter_qwen3_vl_caption_and_relevance(
    *,
    image_path: str,
    description: str,
    keys_file: str,
    logger,
    model_id: str,
    fallback_models: Optional[List[str]] = None,
    max_tokens: int = 220,
    jpeg_quality: int = 75,
    max_attempts=6,
    timeout_s = 5,
min_retry_sleep_s = 0.05

) -> Dict[str, Any]:
    """
    FAST key rotation:
    - No exponential backoff
    - Per-key cooldown only for 429 (rate limit)
    - Immediately disables 401/403 keys
    - Sleeps ONLY when all keys are cooling down
    - Uses requests.Session() keep-alive
    """

    keys = load_keys_from_file(keys_file)
    if not keys:
        return {
            "ok": False,
            "attempt": 0,
            "key_used": "NONE",
            "model": model_id,
            "caption": "",
            "relevant": "NO",
            "recapture": False,
            "reason": "No keys found in keys_file",
            "latency_seconds": 0.0,
            "usage": {},
            "cost": None,
            "image": {},
            "openrouter_id": None,
            "error": {"type": "no_keys"},
        }

    # prepare image once
    data_url, image_meta = image_to_data_url_224(
        image_path,
        size=224,
        quality=max(25, min(95, int(jpeg_quality))),
    )

    # Key states
    disabled = set()  # permanently disabled in this run (401/403)
    cooldown_until: Dict[str, float] = {}  # key -> unix time allowed again

    # session keeps TCP alive (faster)
    session = requests.Session()

    last_error: Optional[Dict[str, Any]] = None

    def _key_ready(k: str) -> bool:
        if k in disabled:
            return False
        t = cooldown_until.get(k)
        if t is None:
            return True
        return time.time() >= t

    def _next_ready_key() -> Optional[str]:
        for k in keys:
            if _key_ready(k):
                return k
        return None

    def _next_wakeup_time() -> Optional[float]:
        future = [t for k, t in cooldown_until.items() if k not in disabled and t > time.time()]
        return min(future) if future else None

    for attempt in range(1, max_attempts + 1):

        key = _next_ready_key()
        if key is None:
            # ✅ all keys cooling down -> sleep until earliest wakeup
            wake = _next_wakeup_time()
            if wake is None:
                # no usable keys at all
                return {
                    "ok": False,
                    "attempt": attempt,
                    "key_used": "NONE_READY",
                    "model": model_id,
                    "caption": "",
                    "relevant": "NO",
                    "recapture": False,
                    "reason": "No usable API keys (all disabled).",
                    "latency_seconds": 0.0,
                    "usage": {},
                    "cost": None,
                    "image": image_meta,
                    "openrouter_id": None,
                    "error": {"type": "no_usable_keys"},
                }

            sleep_s = max(0.0, wake - time.time())
            # keep this short but correct
            sleep_s = max(min_retry_sleep_s, min(sleep_s, 2.0))
            logger.warning(
                "[openrouter_vlm] all_keys_cooldown sleeping=%.3fs attempt=%d/%d",
                sleep_s,
                attempt,
                max_attempts,
            )
            time.sleep(sleep_s)
            continue

        logger.info(
            "[openrouter_vlm] attempt=%d/%d model=%s key=%s img_bytes=%s",
            attempt,
            max_attempts,
            model_id,
            mask_key(key),
            image_meta.get("original_file_bytes"),
        )

        ok, resp = _openrouter_call_once(
            session=session,
            api_key=key,
            model_id=model_id,
            fallback_models=fallback_models,
            image_data_url=data_url,
            description=description or "",
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
                "image": image_meta,
                "openrouter_id": resp.get("openrouter_id"),
            }

            logger.info(
                "[openrouter_vlm] classification relevant=%s recapture=%s unwanted=%s categories=%s reason=%r",
                out["relevant"],
                out["recapture"],
                out["unwanted_content"],
                out["unwanted_categories"],
                out["reason"],
            )
            logger.info(
                "[openrouter_vlm] success attempt=%d relevant=%s recapture=%s unwanted=%s total_tokens=%s cost=%s latency=%.3fs",
                attempt,
                out["relevant"],
                out["recapture"],
                out["unwanted_content"],
                (out["usage"] or {}).get("total_tokens"),
                out.get("cost"),
                float(out.get("latency_seconds") or 0.0),
            )
            return out

        # -------- failure handling (FAST) --------
        last_error = resp
        err_type = resp.get("type")
        http_status = resp.get("http_status")
        retry_after = resp.get("retry_after_seconds")

        logger.warning(
            "[openrouter_vlm] fail attempt=%d key=%s err_type=%s http=%s retry_after=%s",
            attempt,
            mask_key(key),
            err_type,
            http_status,
            retry_after,
        )

        # OpenRouter has already attempted the configured model fallbacks.
        # Retrying an invalid/retired/incompatible model chain will not help.
        if err_type == "http_error" and http_status in (400, 404, 413, 422):
            return {
                "ok": False,
                "attempt": attempt,
                "key_used": mask_key(key),
                "model": model_id,
                "caption": "",
                "relevant": "NO",
                "recapture": False,
                "reason": f"OpenRouter rejected model chain/request with HTTP {http_status}.",
                "latency_seconds": float(resp.get("latency_seconds") or 0.0),
                "usage": {},
                "cost": None,
                "image": image_meta,
                "openrouter_id": None,
                "error": resp,
            }

        # 401/403 => permanently disable key (don’t waste time on it again)
        if err_type == "http_error" and http_status in (401, 403):
            disabled.add(key)
            continue

        # 429 => cooldown this key only, rotate immediately
        if err_type == "http_error" and http_status == 429:
            cooldown_s = float(retry_after) if retry_after is not None else 1.0
            cooldown_until[key] = time.time() + max(0.2, cooldown_s)
            continue

        # provider/server errors => rotate immediately, no sleep
        if err_type in ("provider_error", "missing_choices", "bad_model_output"):
            continue
        if err_type == "http_error" and (http_status and http_status >= 500):
            continue

        # network timeout/connection => rotate immediately, maybe tiny sleep
        if err_type in ("timeout", "request_exception"):
            time.sleep(min_retry_sleep_s)
            continue

        # unknown => rotate immediately
        continue

    return {
        "ok": False,
        "attempt": max_attempts,
        "key_used": "EXHAUSTED",
        "model": model_id,
        "caption": "",
        "relevant": "NO",
        "recapture": False,
        "reason": "All attempts failed (fast key rotation exhausted).",
        "latency_seconds": 0.0,
        "usage": {},
        "cost": None,
        "image": image_meta,
        "openrouter_id": None,
        "error": last_error,
    }