"""Standalone translation API logic (OpenRouter + tencent/hy-mt2-1.8b)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.conf import settings

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Mobicrowd Translation")
DEFAULT_LLM_PROVIDER = "openrouter"
PROVIDER_LABELS = {"openrouter": "OpenRouter"}


def _configured_model(setting_name: str) -> str:
    model = str(getattr(settings, setting_name, "") or "").strip()
    if not model:
        raise RuntimeError(f"{setting_name} is not configured in Django settings.")
    return model


def _configured_fallback_models(
    primary_setting: str,
    fallback_setting: str,
) -> List[str]:
    primary = _configured_model(primary_setting)
    raw = getattr(settings, fallback_setting, []) or []

    if isinstance(raw, str):
        raw = [raw]

    fallbacks: List[str] = []
    for value in raw:
        model = str(value or "").strip()
        if model and model != primary and model not in fallbacks:
            fallbacks.append(model)

    return fallbacks


def _configured_openrouter_keys_file() -> Path:
    raw = str(getattr(settings, "OPENROUTER_KEYS_FILE", "") or "").strip()
    if not raw:
        raise RuntimeError("OPENROUTER_KEYS_FILE is not configured in Django settings.")

    path = Path(raw).expanduser()
    if not path.is_file():
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {path}")
    return path


def _translation_model_chain() -> List[str]:
    primary = _configured_model("OPENROUTER_TRANSLATION_MODEL")
    fallbacks = _configured_fallback_models(
        "OPENROUTER_TRANSLATION_MODEL",
        "OPENROUTER_TRANSLATION_FALLBACK_MODELS",
    )
    return [primary, *fallbacks]


def _detection_model_chain() -> List[str]:
    primary = _configured_model("OPENROUTER_LANGUAGE_DETECTION_MODEL")
    fallbacks = _configured_fallback_models(
        "OPENROUTER_LANGUAGE_DETECTION_MODEL",
        "OPENROUTER_LANGUAGE_DETECTION_FALLBACK_MODELS",
    )
    return [primary, *fallbacks]

_KEY_LOCK = Lock()
_KEY_INDEX = 0
_backend_cache: Dict[str, "OpenRouterBackend"] = {}

# Common aliases → display name for prompts
LANG_ALIASES: Dict[str, str] = {
    "auto": "auto",
    "en": "English",
    "english": "English",
    "fr": "French",
    "french": "French",
    "francais": "French",
    "français": "French",
    "ar": "Arabic",
    "arabic": "Arabic",
    "arabe": "Arabic",
    "arabi": "Arabic",
    "es": "Spanish",
    "spanish": "Spanish",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "russian": "Russian",
    "tr": "Turkish",
    "turkish": "Turkish",
    "nl": "Dutch",
    "dutch": "Dutch",
    "hi": "Hindi",
    "hindi": "Hindi",
}


def _parse_api_keys(raw: str) -> List[str]:
    if not raw:
        return []
    raw = raw.replace(",", "\n")
    keys: List[str] = []
    for line in raw.splitlines():
        key = line.strip()
        if not key or key.startswith("#"):
            continue
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        if "=" in key:
            key = key.split("=", 1)[1].strip()
        if key:
            keys.append(key)
    return keys


def _dedupe_keys(keys: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for key in keys:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def get_openrouter_api_keys() -> List[str]:
    """
    Load OpenRouter credentials only from settings.OPENROUTER_KEYS_FILE.

    No request-level key, helper-local key file, or environment-key fallback
    is accepted.
    """
    key_file = _configured_openrouter_keys_file()
    raw = key_file.read_text(encoding="utf-8")
    keys = _dedupe_keys(_parse_api_keys(raw))

    if not keys:
        raise RuntimeError(
            f"No OpenRouter API key found in configured key file: {key_file}"
        )

    return keys


def _next_api_key(keys: List[str]) -> str:
    global _KEY_INDEX
    if not keys:
        raise RuntimeError("No OpenRouter API keys available.")
    with _KEY_LOCK:
        key = keys[_KEY_INDEX % len(keys)]
        _KEY_INDEX += 1
    return key


def _should_try_next_key(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", status_code)
    return status_code in {401, 402, 403, 429}


def provider_used_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def normalize_language(raw: Any, *, field_name: str, allow_auto: bool = False) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if allow_auto:
            return "auto"
        raise ValueError(f"{field_name} is required.")

    key = str(raw).strip().lower()
    if key in LANG_ALIASES:
        label = LANG_ALIASES[key]
        if label == "auto" and not allow_auto:
            raise ValueError(f"{field_name} cannot be 'auto'.")
        return label

    # Accept free-form language names (e.g. "Tunisian Arabic")
    cleaned = str(raw).strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required.")
    return cleaned


def parse_text(data: Any) -> str:
    text = data.get("text") or data.get("description") or data.get("content") or ""
    text = str(text).strip()
    if not text:
        raise ValueError("text is required (or description / content).")
    return text


def parse_source_lang(data: Any) -> str:
    return normalize_language(
        data.get("source_lang") or data.get("source_language") or data.get("from_lang"),
        field_name="source_lang",
        allow_auto=False,
    )


def parse_target_lang(data: Any) -> str:
    return normalize_language(
        data.get("target_lang") or data.get("target_language") or data.get("to_lang"),
        field_name="target_lang",
        allow_auto=False,
    )


def parse_translate_languages(data: Any) -> Tuple[str, str]:
    source = parse_source_lang(data)
    target = parse_target_lang(data)
    if source.lower() == target.lower():
        raise ValueError("source_lang and target_lang must be different.")
    return source, target


def build_detect_language_prompt(text: str) -> str:
    return (
        "Identify the language of the following text. "
        "Output only the language name in English (e.g. English, French, Arabic). "
        "No explanation or punctuation.\n\n"
        f"{text}"
    )


def build_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    instruction = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        "Output only the translation with no explanation."
    )
    return f"{instruction}\n\n{text}"


def clean_llm_output(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def clean_translation(raw: str) -> str:
    return clean_llm_output(raw)


def clean_detected_language(raw: str) -> str:
    text = clean_llm_output(raw)
    if not text:
        raise ValueError("Language detection returned empty output.")
    text = text.splitlines()[0].strip().rstrip(".")
    return normalize_language(text, field_name="source_lang", allow_auto=False)


class OpenRouterBackend:
    provider = DEFAULT_LLM_PROVIDER
    endpoint = OPENROUTER_ENDPOINT

    def __init__(
        self,
        model_chain: List[str],
        temperature: float = 0.0,
    ):
        if not model_chain:
            raise RuntimeError("OpenRouter model chain cannot be empty.")

        self.model_chain = list(model_chain)
        self.model_name = self.model_chain[0]
        self.fallback_models = self.model_chain[1:]
        self.temperature = temperature
        self.api_keys = get_openrouter_api_keys()

    def chat(self, user_message: str, max_tokens: int = 2048) -> str:
        last_error: Optional[Exception] = None
        attempts = max(1, len(self.api_keys))

        for attempt in range(attempts):
            api_key = _next_api_key(self.api_keys)
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": OPENROUTER_REFERER,
                        "X-OpenRouter-Title": OPENROUTER_TITLE,
                    },
                    json={
                        "models": self.model_chain,
                        "messages": [{"role": "user", "content": user_message}],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                        "provider": {"allow_fallbacks": True},
                    },
                    timeout=(5.0, 120.0),
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                return str(message.get("content") or "").strip()
            except requests.HTTPError as exc:
                if attempt < attempts - 1 and _should_try_next_key(exc):
                    continue
                status_code = exc.response.status_code if exc.response is not None else 0
                raise RuntimeError(
                    f"{status_code} OpenRouter error for {self.endpoint} "
                    f"(tried {attempt + 1}/{attempts} key(s))"
                ) from exc
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1 and _should_try_next_key(exc):
                    continue
                raise

        if last_error:
            raise last_error
        return ""


def _get_backend(role: str) -> OpenRouterBackend:
    if role == "translate":
        model_chain = _translation_model_chain()
    elif role == "detect":
        model_chain = _detection_model_chain()
    else:
        raise ValueError(f"Unsupported OpenRouter translation role: {role}")

    keys_file = str(_configured_openrouter_keys_file())
    cache_key = f"{role}|{'|'.join(model_chain)}|{keys_file}|0.0"

    if cache_key not in _backend_cache:
        _backend_cache[cache_key] = OpenRouterBackend(
            model_chain=model_chain,
            temperature=0.0,
        )

    return _backend_cache[cache_key]


def run_detect_language(
    text: str,
) -> Dict[str, Any]:
    backend = _get_backend("detect")
    prompt = build_detect_language_prompt(text)
    raw = backend.chat(prompt, max_tokens=64)
    try:
        source_lang = clean_detected_language(raw)
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "text": text,
            "source_lang": "",
            "model": backend.model_name,
            "configured_fallback_models": backend.fallback_models,
            "provider_used": backend.provider,
            "openrouter_key_count": len(backend.api_keys),
        }

    return {
        "ok": True,
        "text": text,
        "source_lang": source_lang,
        "model": backend.model_name,
        "configured_fallback_models": backend.fallback_models,
        "provider_used": backend.provider,
        "openrouter_key_count": len(backend.api_keys),
        "error": None,
    }


def run_translate(
    text: str,
    *,
    source_lang: str,
    target_lang: str,
) -> Dict[str, Any]:
    backend = _get_backend("translate")
    prompt = build_translation_prompt(text, source_lang, target_lang)
    raw = backend.chat(prompt)
    translated = clean_translation(raw)
    if not translated:
        return {
            "ok": False,
            "error": "Translation returned empty output.",
            "text": text,
            "translated_text": "",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": backend.model_name,
            "configured_fallback_models": backend.fallback_models,
            "provider_used": backend.provider,
            "openrouter_key_count": len(backend.api_keys),
        }

    return {
        "ok": True,
        "text": text,
        "translated_text": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": backend.model_name,
        "configured_fallback_models": backend.fallback_models,
        "provider_used": backend.provider,
        "openrouter_key_count": len(backend.api_keys),
        "error": None,
    }