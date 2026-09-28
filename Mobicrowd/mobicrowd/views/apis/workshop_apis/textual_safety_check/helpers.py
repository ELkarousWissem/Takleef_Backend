import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests
from django.conf import settings

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


OPENROUTER_HTTP_REFERER = "http://localhost"
OPENROUTER_APP_TITLE = "Mobicrowd Textual Safety Check"


def _configured_textual_safety_model() -> str:
    """Primary textual-safety model from Django settings only."""
    model = str(
        getattr(settings, "OPENROUTER_TEXTUAL_SAFETY_MODEL", "") or ""
    ).strip()

    if not model:
        raise RuntimeError(
            "OPENROUTER_TEXTUAL_SAFETY_MODEL is not configured in Django settings."
        )

    return model


def _configured_textual_safety_fallback_models() -> List[str]:
    """Ordered textual-safety fallback models from Django settings only."""
    primary = _configured_textual_safety_model()
    raw = getattr(
        settings,
        "OPENROUTER_TEXTUAL_SAFETY_FALLBACK_MODELS",
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
_OPENROUTER_KEY_INDEX = 0
_OPENROUTER_SESSION: Optional[requests.Session] = None

UNSAFE_CATEGORY_TERMS: Dict[str, List[str]] = {
    "racism": [
        "racial slur",
        "white power",
        "nazi",
        "supremacy",
        "hate race",
        "عنصري",
        "كراهية عرقية",
        "تفوق عرقي",
    ],
    "sexual": [
        "sex",
        "sexual",
        "nude",
        "porn",
        "explicit",
        "fetish",
        "إيحاء جنسي",
        "جنس",
        "إباحي",
        "عري",
    ],
    "violence": [
        "kill",
        "murder",
        "shoot",
        "stab",
        "behead",
        "bomb",
        "terror",
        "قتل",
        "ذبح",
        "تفجير",
        "إرهاب",
        "عنف",
    ],
    "profanity": [
        "fuck",
        "shit",
        "asshole",
        "bastard",
        "قذر",
        "شتيمة",
        "كلام بذيء",
    ],
}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_arabic_text(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in (text or ""))


def load_openrouter_keys() -> List[str]:
    """
    Load OpenRouter credentials exclusively from settings.OPENROUTER_KEYS_FILE.

    No helper-local key file, alternate key file, legacy provider key file,
    or environment-key fallback is accepted.
    """
    key_file = _configured_openrouter_keys_file()
    keys: List[str] = []

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
    global _OPENROUTER_KEY_INDEX

    keys = load_openrouter_keys()

    _OPENROUTER_KEY_LOCK.acquire()

    try:
        key = keys[_OPENROUTER_KEY_INDEX % len(keys)]
        _OPENROUTER_KEY_INDEX += 1
    finally:
        _OPENROUTER_KEY_LOCK.release()

    return key.strip()



def _get_openrouter_session() -> requests.Session:
    global _OPENROUTER_SESSION
    if _OPENROUTER_SESSION is None:
        _OPENROUTER_SESSION = requests.Session()
    return _OPENROUTER_SESSION


def _should_try_next_key(status_code: Optional[int], exc: Optional[Exception] = None) -> bool:
    if status_code in {401, 403, 429, 502, 503}:
        return True
    if exc is None:
        return False
    msg = str(exc).lower()
    markers = (
        "rate limit", "rate_limit", "invalid api key", "authentication",
        "unauthorized", "quota", "insufficient", "429", "401", "403",
        "timeout", "connection",
    )
    return any(marker in msg for marker in markers)


def openrouter_chat_completion(
    *,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 250,
    max_attempts: Optional[int] = None,
) -> str:
    primary_model = _configured_textual_safety_model()
    fallback_models = _configured_textual_safety_fallback_models()
    model_chain = [primary_model, *fallback_models]

    keys = load_openrouter_keys()
    session = _get_openrouter_session()
    attempts = max_attempts or len(keys)
    last_error: Any = None

    for attempt in range(attempts):
        api_key = keys[attempt % len(keys)]
        payload = {
            "models": model_chain,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "provider": {"allow_fallbacks": True},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_APP_TITLE,
        }
        try:
            response = session.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=(5.0, 60.0),
            )
            data = response.json() if response.content else {}
            if response.status_code == 200 and data.get("choices"):
                return (data["choices"][0].get("message", {}).get("content") or "").strip()
            last_error = RuntimeError(f"OpenRouter HTTP {response.status_code}: {data}")
            if attempt < attempts - 1 and _should_try_next_key(response.status_code):
                continue
            raise last_error
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1 and _should_try_next_key(None, exc):
                continue
            raise

    raise RuntimeError(f"OpenRouter chat completion failed: {last_error}")

def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if not text:
        return {}

    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])

            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def detect_unsafe_keywords(text: str) -> Dict[str, Any]:
    lowered = (text or "").lower()

    found_categories = []
    found_terms = []

    for category, terms in UNSAFE_CATEGORY_TERMS.items():
        matched_terms = [term for term in terms if term in lowered]

        if matched_terms:
            found_categories.append(category)
            found_terms.extend(matched_terms[:3])

    return {
        "detected": bool(found_categories),
        "categories": found_categories,
        "terms": found_terms[:8],
    }


def llm_safety_review(
    text: str,
    language: str,
) -> Dict[str, Any]:
    system_prompt = (
        "You classify whether text is safe for work in a professional crowdsourcing platform. "
        "Mark unsafe for hate speech, racism, sexual content, graphic violence, threats, "
        "harassment, or strong profanity. Neutral opinions and mild language are safe for work."
    )

    reason_language = "Arabic" if language == "ar" else "English"

    user_prompt = f"""
Text to classify:
{text}

Return ONLY valid JSON:
{{
  "safe_for_work": "YES or NO",
  "confidence": 0.0,
  "categories": ["racism|sexual|violence|profanity|harassment|threats|other"],
  "reason": "one short sentence"
}}

Rules:
- safe_for_work=YES unless clearly unsafe.
- Do not flag professional discussion without harmful intent.
- Write reason in {reason_language}.
""".strip()

    try:
        raw_response = openrouter_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )

        parsed = extract_json(raw_response)
        safe_raw = str(parsed.get("safe_for_work", "YES")).strip().upper()

        is_safe = safe_raw == "YES"

        try:
            confidence = float(parsed.get("confidence", 0.9 if is_safe else 0.85))
        except Exception:
            confidence = 0.9 if is_safe else 0.85

        confidence = max(0.0, min(1.0, confidence))

        categories = parsed.get("categories", [])

        if not isinstance(categories, list):
            categories = []

        categories = [str(category).strip() for category in categories if str(category).strip()]

        reason = str(parsed.get("reason", "")).strip()

        return {
            "is_safe_for_work": is_safe,
            "confidence": confidence,
            "categories": categories,
            "reason": reason,
            "raw_response": raw_response,
        }

    except Exception as e:
        return {
            "is_safe_for_work": True,
            "confidence": 0.5,
            "categories": [],
            "reason": f"LLM safety check failed; defaulted to safe. ({e})",
            "raw_response": "",
            "llm_error": str(e),
        }


def check_text_safety(
    text: str,
    use_llm: bool = True,
    output_language: str = "auto",
) -> Dict[str, Any]:
    effective_model = _configured_textual_safety_model()

    text = (text or "").strip()

    if not text:
        return {
            "ok": False,
            "status": "error",
            "error": "Empty text",
            "message": "Empty text",
        }

    language = (output_language or "auto").strip().lower()

    if language == "auto":
        language = "ar" if is_arabic_text(text) else "en"

    keyword_hit = detect_unsafe_keywords(text)

    if keyword_hit.get("detected"):
        categories = keyword_hit.get("categories") or []
        category_text = ", ".join(categories) or "unsafe content"

        return {
            "ok": True,
            "status": "success",
            "is_safe_for_work": False,
            "safe": "NO",
            "score": 0.0,
            "reason": f"Blocked by keyword safety gate ({category_text}).",
            "categories": categories,
            "matched_terms": keyword_hit.get("terms") or [],
            "method": "keyword",
            "unsafe_content": keyword_hit,
            "model": effective_model,
            "raw_response": "blocked_by_keyword_gate",
        }

    if not use_llm:
        return {
            "ok": True,
            "status": "success",
            "is_safe_for_work": True,
            "safe": "YES",
            "score": 1.0,
            "reason": "No unsafe keywords detected.",
            "categories": [],
            "matched_terms": [],
            "method": "keyword",
            "unsafe_content": keyword_hit,
            "model": effective_model,
            "raw_response": "",
        }

    llm_result = llm_safety_review(
        text=text,
        language=language,
    )
    is_safe = bool(llm_result.get("is_safe_for_work", True))
    categories = list(llm_result.get("categories") or [])
    reason = str(llm_result.get("reason", "")).strip()

    if not is_safe and not reason:
        reason = "Marked not safe for work by content review."

    if is_safe and not reason:
        reason = "Safe for work: no harmful content detected."

    unsafe_content = {
        "detected": not is_safe,
        "categories": categories,
        "terms": keyword_hit.get("terms") or [],
    }

    return {
        "ok": True,
        "status": "success",
        "is_safe_for_work": is_safe,
        "safe": "YES" if is_safe else "NO",
        "score": float(llm_result.get("confidence", 1.0 if is_safe else 0.0)),
        "reason": reason,
        "categories": categories,
        "matched_terms": keyword_hit.get("terms") or [],
        "method": "keyword+llm",
        "unsafe_content": unsafe_content,
        "model": effective_model,
        "raw_response": llm_result.get("raw_response", ""),
        "llm_error": llm_result.get("llm_error"),
    }