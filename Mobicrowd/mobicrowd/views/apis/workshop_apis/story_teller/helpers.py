import base64
import json
import os
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from openai import OpenAI


IMAGE_BATCH_SIZE = 4
IMAGE_MAX_WIDTH = 640
VIDEO_FRAME_COUNT = 8
VIDEO_FRAME_MAX_WIDTH = 320
JPEG_QUALITY = 82

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

MAX_IMAGE_SIZE_MB = 15
MAX_VIDEO_SIZE_MB = 250

_OPENROUTER_KEY_FILE = Path(settings.OPENROUTER_KEYS_FILE)
_OPENROUTER_KEY_FILE_ALT = _OPENROUTER_KEY_FILE
_OPENROUTER_KEY_LOCK = Lock()
_OPENROUTER_KEY_INDEX = int.from_bytes(os.urandom(2), "big")


STORY_SYSTEM_PROMPT = (
    "You are Story Teller—a sharp visual narrator for a media workshop. "
    "Your job is to turn images and video frames into text that is accurate, ordered, and readable. "
    "Rules you always follow:\n"
    "1) Ground every claim in what is visible; do not invent people, places, or events that are not supported.\n"
    "2) Use aliases only (Image 1, Image 2, Video 1, …)—never raw filenames or file paths.\n"
    "3) Respect the media index order; that order is the timeline of the collection.\n"
    "4) Write clear, confident prose—no analyst jargon, no filler, no report-style section titles.\n"
    "5) Output exactly one format: the selected mode below—never mix modes or add extra sections."
)

REDUCE_SYSTEM_ADDON = (
    " You are in the MERGE step: intermediate captions are evidence, not the final draft. "
    "Synthesize them into one polished answer for the selected mode. "
    "Drop duplicate facts, fix contradictions by trusting visible evidence, and keep the voice consistent."
)

MAP_IMAGE_SYSTEM = (
    "You are the OBSERVATION step for still images. "
    "Record only what each image shows—subject, action, setting, lighting, and mood. "
    "Be precise and neutral; this text will be merged later. Use only the aliases in this batch."
)

MAP_VIDEO_SYSTEM = (
    "You are the OBSERVATION step for one video. "
    "Frames are stratified samples in time order. Describe how the scene evolves: "
    "opening situation → key actions or changes → end state. "
    "Use the video alias; cite [@ seconds] only when timing clarifies the action."
)

FOLLOWUP_SYSTEM = (
    "You are Story Teller's chat assistant. "
    "Help the user with the generated story and media aliases (Image 1, Video 2, …). "
    "You MAY: answer questions about the story, rewrite or restyle it "
    "(happy, formal, shorter, caption, CTA, etc.), compare images, or clarify details. "
    "Stay grounded in the story; do not invent new visual facts. "
    "If something is not in the story, say so briefly. "
    "OUTPUT RULES (critical): "
    "Return ONLY the user-facing reply. "
    "Never output thinking, analysis, planning, step lists, constraints, "
    "or phrases like 'Analyze the Content', 'Evaluate Support', 'Final Polish'."
)

# Vision/story generation model (multimodal)
STORY_MODEL = settings.OPENROUTER_STORY_TELLER_MODEL
# Follow-up is text-only — use a non-reasoning chat model to avoid CoT leaks
FOLLOWUP_MODEL = settings.OPENROUTER_STORY_TELLER_FALLBACK_MODELS


MODE_INSTRUCTIONS: Dict[str, str] = {
    "detailed_summary": (
        "DETAILED SUMMARY — Cover every item in the media index, in order.\n"
        "For each item: open with its alias on its own line, then 2–4 sentences covering "
        "who/what is visible, what they are doing, where it happens, and the overall mood or lighting. "
        "For videos, mention progression across time when known. One item = one block; do not skip items."
    ),
    "short_summary": (
        "SHORT SUMMARY — Cluster items by visual similarity: subject, setting, activity, or mood.\n"
        "Label each cluster with a short heading. Under each heading, write one tight paragraph that summarizes "
        "all items in that cluster using their aliases. Every alias from the index must appear exactly once."
    ),
    "story": (
        "STORY — One chronological narrative in 2–4 short paragraphs, following the media index order.\n"
        "Treat the collection as a single journey: who or what we follow, what changes from scene to scene, "
        "and why the sequence matters. Use vivid but honest language grounded in pixels."
    ),
}

MODE_LABEL_TO_VALUE = {
    "Detailed summary (caption each image)": "detailed_summary",
    "Short summary (group by similarity)": "short_summary",
    "Story (chronological relation)": "story",
}

MODE_VALUE_TO_LABEL = {value: key for key, value in MODE_LABEL_TO_VALUE.items()}

MAP_HINTS = {
    "detailed_summary": "Capture facts needed for a per-item caption later.",
    "short_summary": "Note cluster cues: shared setting, subject type, or mood.",
    "story": "Note narrative beats: opening hook, change, mood shift, closing impression.",
}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: Any, default: int, min_value: int = 1, max_value: int = 5000) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default

    return max(min_value, min(parsed, max_value))


def load_openrouter_keys() -> List[str]:
    keys: List[str] = []
    seen = set()

    for path in (_OPENROUTER_KEY_FILE, _OPENROUTER_KEY_FILE_ALT):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if not key or key.startswith("#"):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)

    if not keys:
        env_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        if env_key:
            keys = [env_key]

    if not keys:
        raise RuntimeError(
            f"OpenRouter key not found. Add openrouter-key.txt next to helpers.py "
            f"or set OPENROUTER_API_KEY."
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


def is_openrouter_key_failure(error: Exception) -> bool:
    message = str(error).lower()

    key_failure_markers = (
        "401",
        "402",
        "403",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "insufficient credits",
        "credits",
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
    )

    return any(marker in message for marker in key_failure_markers)


def is_token_fallback_error(error: Exception) -> bool:
    message = str(error).lower()

    token_markers = (
        "fewer max_tokens",
        "max_tokens",
        "maximum context",
        "context length",
        "context_length",
        "token limit",
        "too many tokens",
    )

    return any(marker in message for marker in token_markers)


def _extract_message_text(message: Any) -> str:
    """Pull assistant text from OpenAI-compatible message objects/dicts."""
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content or "").strip()


def looks_like_reasoning_dump(text: str) -> bool:
    """Detect chain-of-thought / planning text that should never be shown to users."""
    s = (text or "").strip().lower()
    if not s:
        return False

    markers = (
        "thinking process",
        "analyze the request",
        "analyze the content",
        "evaluate support",
        "evaluate available",
        "formulate the",
        "drafting the",
        "final polish",
        "selected response",
        "constraint 1",
        "constraint 2",
        "generated story:",
        "user question:",
        "look at the constraint",
        "does the story support",
        "internal monologue",
    )
    hits = sum(1 for m in markers if m in s)
    numbered_steps = len(re.findall(r"(?m)^\s*\d+\.\s+\*?\*?[a-z]", text or ""))
    if hits >= 2:
        return True
    if numbered_steps >= 2 and hits >= 1:
        return True
    if re.match(r"^[\s.]*\d+\.\s+\*?\*?analyze\b", s):
        return True
    if s.startswith(("thinking process", "1.  **analyze", ". 2. **analyze", "2. **analyze")):
        return True
    return False


def strip_model_reasoning(text: str) -> str:
    """
    Qwen-style models often dump internal planning into content.
    Keep only the user-facing answer.
    """
    s = (text or "").strip()
    if not s:
        return ""

    for marker in ("</think>", "</thinking>", "</reasoning>"):
        if marker in s:
            after = s.split(marker)[-1].strip()
            if after and not looks_like_reasoning_dump(after):
                return after
            if after:
                s = after
                break

    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<thinking>[\s\S]*?</thinking>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", s, flags=re.IGNORECASE)

    finals = list(
        re.finditer(
            r"(?is)\b(?:final\s+answer|final\s+response|user[- ]facing\s+reply)\s*:\s*",
            s,
        )
    )
    if finals:
        candidate = s[finals[-1].end() :].strip().strip('"')
        candidate = re.split(
            r"(?is)\n\s*(?:\d+\.\s+)?\*?\*?(?:analyze|evaluate|formulate|drafting|thinking)\b",
            candidate,
            maxsplit=1,
        )[0].strip()
        if candidate and not looks_like_reasoning_dump(candidate):
            return candidate

    if looks_like_reasoning_dump(s) or re.search(
        r"(?is)(analyze the (?:request|content)|evaluate support|thinking process)",
        s,
    ):
        return ""

    cleaned_lines: List[str] = []
    for line in s.splitlines():
        lower = line.strip().lower()
        if not lower:
            cleaned_lines.append(line)
            continue
        if re.match(r"^\d+\.\s+", lower) and any(
            k in lower
            for k in ("analyze", "evaluate", "formulate", "draft", "polish", "constraint")
        ):
            continue
        if lower.startswith(
            ("constraint", "role:", "task:", "selected response:", "*selected", "user question:")
        ):
            continue
        if "answer from the generated story" in lower:
            continue
        cleaned_lines.append(line)

    s = "\n".join(cleaned_lines).strip()
    s = re.sub(r"^\*+Selected Response:\*+\s*", "", s, flags=re.IGNORECASE).strip()
    if looks_like_reasoning_dump(s):
        return ""
    if re.match(r"(?is)^\d+\.\s+\*?\*?[a-z ].{0,80}$", s):
        return ""
    return s


def strip_mode_instruction_echo(text: str, output_mode: str = "") -> str:
    """Remove leaked mode-instruction prompts from story output."""
    s = (text or "").strip()
    if not s:
        return ""

    mode = normalize_output_mode(output_mode) if output_mode else ""
    instruction_blobs = list(MODE_INSTRUCTIONS.values())
    if mode and mode in MODE_INSTRUCTIONS:
        instruction_blobs = [MODE_INSTRUCTIONS[mode]] + [
            v for k, v in MODE_INSTRUCTIONS.items() if k != mode
        ]

    for blob in instruction_blobs:
        first_line = blob.strip().splitlines()[0].strip()
        # Exact prefix echo of the mode contract
        if s.startswith(blob.strip()):
            s = s[len(blob.strip()):].lstrip(" \n-")
            break
        if first_line and s.startswith(first_line):
            # Drop until first blank line after the echoed contract
            rest = s[len(first_line):]
            # If more of the instruction follows, cut at first Image/Video alias block
            m = re.search(r"(?m)^(Image\s+\d+|Video\s+\d+)\b", s)
            if m and m.start() > 0:
                s = s[m.start():].strip()
                break
            # Otherwise drop the first paragraph
            parts = re.split(r"\n\s*\n", s, maxsplit=1)
            if len(parts) == 2 and len(parts[0]) < 500:
                s = parts[1].strip()
                break

    # Generic heading echoes
    s = re.sub(
        r"(?is)^\s*(DETAILED SUMMARY|SHORT SUMMARY|STORY)\s*[—\-].*?(?=\n\s*\n|\nImage\s+\d+|\nVideo\s+\d+)",
        "",
        s,
        count=1,
    ).strip()
    return s


def clean_story_output(text: str, output_mode: str = "") -> str:
    return strip_mode_instruction_echo(strip_model_reasoning(text), output_mode=output_mode)


def story_teller_chat_completion(
    messages: List[Dict[str, Any]],
    max_tokens: int = 1200,
    model: Optional[str] = None,
) -> str:
    model_id = (model or STORY_MODEL).strip() or STORY_MODEL
    attempts = []

    for token_count in [max_tokens, 1200, 900, 700, 500, 350, 220]:
        if token_count not in attempts:
            attempts.append(token_count)

    keys = load_openrouter_keys()
    last_error: Optional[Exception] = None

    for _ in range(len(keys)):
        api_key = resolve_openrouter_api_key()

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        for token_count in attempts:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=token_count,
                    temperature=0.35,
                    top_p=0.9,
                )

                raw = _extract_message_text(
                    response.choices[0].message if response.choices else None
                )
                cleaned = strip_model_reasoning(raw)
                if cleaned:
                    return cleaned
                # Never return raw reasoning dumps to the UI
                if raw and not looks_like_reasoning_dump(raw):
                    return raw
                return ""

            except Exception as e:
                last_error = e

                if is_openrouter_key_failure(e):
                    break

                if is_token_fallback_error(e):
                    continue

                raise

    raise RuntimeError(
        f"Story Teller failed after trying {len(keys)} key(s) "
        f"and token fallbacks {attempts}: {last_error}"
    )


def normalize_output_mode(output_mode: str) -> str:
    mode = str(output_mode or "detailed_summary").strip().lower()

    legacy_map = {
        "summary": "detailed_summary",
        "both": "short_summary",
    }

    mode = legacy_map.get(mode, mode)

    if mode not in MODE_INSTRUCTIONS:
        mode = MODE_LABEL_TO_VALUE.get(output_mode, "detailed_summary")

    if mode not in MODE_INSTRUCTIONS:
        mode = "detailed_summary"

    return mode


def mode_to_display_label(output_mode: str) -> str:
    mode = normalize_output_mode(output_mode)
    return MODE_VALUE_TO_LABEL.get(mode, "Detailed summary (caption each image)")


def is_video_path(path: str) -> bool:
    return Path(path or "").suffix.lower() in VIDEO_EXTENSIONS


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]

    return [items[index : index + size] for index in range(0, len(items), size)]


def stratified_frame_indices(
    total_frames: int,
    count: int = VIDEO_FRAME_COUNT,
) -> List[int]:
    if total_frames <= 0:
        return []

    if total_frames <= count:
        return list(range(total_frames))

    indices = np.linspace(0, total_frames - 1, count)

    return sorted({int(round(index)) for index in indices})


def resize_bgr(
    bgr: np.ndarray,
    max_width: int,
) -> np.ndarray:
    height, width = bgr.shape[:2]

    if width <= max_width:
        return bgr

    scale = max_width / float(width)
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        bgr,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def bgr_to_jpeg_base64(
    bgr: np.ndarray,
    max_width: int,
) -> str:
    resized = resize_bgr(bgr, max_width)

    ok, encoded = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )

    if not ok:
        raise ValueError("Failed to encode JPEG.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def image_url_part(
    image_base64: str,
    mime: str = "image/jpeg",
) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{image_base64}",
        },
    }


def validate_uploaded_media_file(uploaded_file: UploadedFile) -> None:
    if uploaded_file is None:
        raise ValueError("Media file is required.")

    filename = getattr(uploaded_file, "name", "")
    suffix = Path(filename).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        max_size_mb = MAX_IMAGE_SIZE_MB
    elif suffix in VIDEO_EXTENSIONS:
        max_size_mb = MAX_VIDEO_SIZE_MB
    else:
        content_type = str(getattr(uploaded_file, "content_type", "") or "").lower()

        if content_type.startswith("image/"):
            max_size_mb = MAX_IMAGE_SIZE_MB
        elif content_type.startswith("video/"):
            max_size_mb = MAX_VIDEO_SIZE_MB
        else:
            raise ValueError(
                "Unsupported media type. Use image or video files."
            )

    max_size = max_size_mb * 1024 * 1024

    if uploaded_file.size > max_size:
        raise ValueError(f"File too large. Maximum size is {max_size_mb} MB.")


def save_uploaded_media_to_temp(uploaded_file: UploadedFile) -> str:
    validate_uploaded_media_file(uploaded_file)

    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()

    if not suffix:
        content_type = str(getattr(uploaded_file, "content_type", "") or "").lower()

        if content_type.startswith("video/"):
            suffix = ".mp4"
        else:
            suffix = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)

        return temp_file.name


def save_uploaded_media_files_to_temp(uploaded_files: List[UploadedFile]) -> List[str]:
    if not uploaded_files:
        raise ValueError("At least one media file is required.")

    temp_paths = []

    for uploaded_file in uploaded_files:
        temp_paths.append(save_uploaded_media_to_temp(uploaded_file))

    return temp_paths


def remove_temp_files(paths: List[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except Exception:
            pass


def story_alias_metadata(media_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    image_index = 0
    video_index = 0

    for item in media_metadata or []:
        kind = str(item.get("kind", "image")).strip().lower()

        if kind == "video":
            video_index += 1
            alias = f"Video {video_index}"
        else:
            image_index += 1
            alias = f"Image {image_index}"
            kind = "image"

        output.append(
            {
                "alias": alias,
                "kind": kind,
            }
        )

    return output


def build_story_media_metadata(file_paths: List[str]) -> List[Dict[str, Any]]:
    metadata = []

    for index, path in enumerate(file_paths or []):
        if not path:
            continue

        kind = "video" if is_video_path(path) else "image"

        item = {
            "id": f"media_{index + 1}",
            "kind": kind,
        }

        try:
            item["size_bytes"] = int(os.path.getsize(path))
        except Exception:
            pass

        metadata.append(item)

    return metadata


def plan_media_units(file_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    paths = [path for path in file_paths if path]

    metadata = build_story_media_metadata(paths)
    aliases = story_alias_metadata(metadata)

    images = []
    videos = []

    for order, (entry, path) in enumerate(zip(aliases, paths)):
        unit = {
            **entry,
            "path": path,
            "order": order,
        }

        if entry.get("kind") == "video":
            videos.append(unit)
        else:
            images.append(unit)

    return images, videos


def load_image_base64(path: str) -> str:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Unreadable image.")

    return bgr_to_jpeg_base64(bgr, IMAGE_MAX_WIDTH)


def load_video_frames_base64(path: str) -> List[Tuple[float, str]]:
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError("Failed to open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if fps <= 0:
        fps = 25.0

    indices = stratified_frame_indices(total_frames if total_frames > 0 else 1)

    frames = []

    if not indices:
        indices = [0]

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        timestamp = index / fps
        frame_base64 = bgr_to_jpeg_base64(frame, VIDEO_FRAME_MAX_WIDTH)

        frames.append((timestamp, frame_base64))

    cap.release()

    if not frames:
        raise ValueError("Could not read video frames.")

    return frames


def build_story_user_prompt(
    output_mode: str,
    media_metadata: List[Dict[str, Any]],
    recap_prompt: str = "",
    original_request: str = "",
    intermediate_notes: str = "",
) -> str:
    mode = normalize_output_mode(output_mode)

    parts = [
        f"Original request:\n{(original_request or '').strip() or '[None]'}\n",
        f"Optional recap prompt:\n{(recap_prompt or '').strip() or '[None]'}\n",
        f"Selected output mode: {mode}\n",
        f"Full media index in upload order:\n{json.dumps(media_metadata, ensure_ascii=False)}\n",
    ]

    if intermediate_notes.strip():
        parts.append(
            "Intermediate captions from sub-generations:\n"
            f"{intermediate_notes.strip()}\n"
        )

    parts.append(
        "Final output contract:\n"
        f"{MODE_INSTRUCTIONS[mode]}\n"
        "- Deliver only this mode.\n"
        "- Do NOT reprint these instructions or the mode title in your answer.\n"
        "- Start directly with the story/captions (e.g. Image 1 / cluster headings).\n"
        "- Honor upload order from the media index.\n"
        "- If recap prompt conflicts with visible evidence, follow the evidence.\n"
        "- Do not print filenames or raw file paths."
    )

    return "\n".join(parts)


def reduce_system_for_mode(output_mode: str) -> str:
    mode = normalize_output_mode(output_mode)

    return (
        STORY_SYSTEM_PROMPT
        + REDUCE_SYSTEM_ADDON
        + f"\n\nTarget mode:\n{MODE_INSTRUCTIONS[mode]}"
    )


def map_image_batch(
    batch: List[Dict[str, Any]],
    recap_prompt: str,
    output_mode: str,
) -> str:
    mode = normalize_output_mode(output_mode)
    aliases = [unit["alias"] for unit in batch]

    user_content = [
        {
            "type": "text",
            "text": (
                f"Observe each still image in this batch. Final mode: {mode}.\n"
                f"Aliases in order: {', '.join(aliases)}\n"
                f"User focus: {(recap_prompt or '').strip() or '[None]'}\n"
                f"Hint for merge step: {MAP_HINTS[mode]}\n\n"
                "For each alias, write a short observation block with visible facts only."
            ),
        }
    ]

    for unit in batch:
        image_base64 = load_image_base64(unit["path"])

        user_content.append(
            {
                "type": "text",
                "text": f"[{unit['alias']}]",
            }
        )

        user_content.append(image_url_part(image_base64))

    return story_teller_chat_completion(
        [
            {
                "role": "system",
                "content": MAP_IMAGE_SYSTEM,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        max_tokens=900,
    )


def map_video_unit(
    unit: Dict[str, Any],
    recap_prompt: str,
    output_mode: str,
) -> str:
    mode = normalize_output_mode(output_mode)
    alias = unit["alias"]

    frames = load_video_frames_base64(unit["path"])

    user_content = [
        {
            "type": "text",
            "text": (
                f"Observe {alias} from {len(frames)} stratified frames in chronological order. "
                f"Final mode: {mode}.\n"
                f"User focus: {(recap_prompt or '').strip() or '[None]'}\n"
                f"Hint for merge step: {MAP_HINTS[mode]}\n\n"
                f"Write one observation block for {alias}: start → middle → end. "
                "Use visible evidence only."
            ),
        }
    ]

    for timestamp, image_base64 in frames:
        user_content.append(
            {
                "type": "text",
                "text": f"[{alias} @ {timestamp:.1f}s]",
            }
        )

        user_content.append(image_url_part(image_base64))

    return story_teller_chat_completion(
        [
            {
                "role": "system",
                "content": MAP_VIDEO_SYSTEM,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        max_tokens=900,
    )


def needs_map_reduce(
    image_units: List[Dict[str, Any]],
    video_units: List[Dict[str, Any]],
) -> bool:
    if video_units:
        return True

    return len(image_units) > IMAGE_BATCH_SIZE


def run_map_phase(
    image_units: List[Dict[str, Any]],
    video_units: List[Dict[str, Any]],
    recap_prompt: str,
    output_mode: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    partials = []

    stats = {
        "image_batches": 0,
        "videos_processed": 0,
        "images_processed": len(image_units),
    }

    for batch in chunk_list(image_units, IMAGE_BATCH_SIZE):
        label = f"{batch[0]['alias']}–{batch[-1]['alias']}"

        text = map_image_batch(
            batch=batch,
            recap_prompt=recap_prompt,
            output_mode=output_mode,
        )

        partials.append(
            {
                "label": label,
                "kind": "image_batch",
                "text": text,
            }
        )

        stats["image_batches"] += 1

    for unit in video_units:
        text = map_video_unit(
            unit=unit,
            recap_prompt=recap_prompt,
            output_mode=output_mode,
        )

        partials.append(
            {
                "label": unit["alias"],
                "kind": "video",
                "text": text,
            }
        )

        stats["videos_processed"] += 1

    return partials, stats


def partials_to_notes(partials: List[Dict[str, str]]) -> str:
    blocks = []

    for partial in partials:
        blocks.append(
            f"### {partial['label']}\n{partial['text']}"
        )

    return "\n\n".join(blocks)


def single_shot_images(
    image_units: List[Dict[str, Any]],
    media_metadata: List[Dict[str, Any]],
    output_mode: str,
    recap_prompt: str,
    original_request: str,
) -> str:
    mode = normalize_output_mode(output_mode)

    user_content = [
        {
            "type": "text",
            "text": build_story_user_prompt(
                output_mode=mode,
                media_metadata=media_metadata,
                recap_prompt=recap_prompt,
                original_request=original_request,
            ),
        }
    ]

    for unit in image_units:
        image_base64 = load_image_base64(unit["path"])

        user_content.append(
            {
                "type": "text",
                "text": f"[{unit['alias']}]",
            }
        )

        user_content.append(image_url_part(image_base64))

    system = STORY_SYSTEM_PROMPT + f"\n\nApply this mode now:\n{MODE_INSTRUCTIONS[mode]}"

    return story_teller_chat_completion(
        [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        max_tokens=1200,
    )


def reduce_phase(
    partials: List[Dict[str, str]],
    media_metadata: List[Dict[str, Any]],
    output_mode: str,
    recap_prompt: str,
    original_request: str,
) -> str:
    notes = partials_to_notes(partials)

    user_text = build_story_user_prompt(
        output_mode=output_mode,
        media_metadata=media_metadata,
        recap_prompt=recap_prompt,
        original_request=original_request,
        intermediate_notes=notes,
    )

    return story_teller_chat_completion(
        [
            {
                "role": "system",
                "content": reduce_system_for_mode(output_mode),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        max_tokens=1400,
    )


def generate_story_teller_output(
    media_file_paths: List[str],
    recap_prompt: str = "",
    output_mode: str = "detailed_summary",
    original_request: str = "",
) -> Dict[str, Any]:
    if not media_file_paths:
        return {
            "status": "error",
            "message": "Select at least one media file.",
        }

    output_mode = normalize_output_mode(output_mode)

    image_units, video_units = plan_media_units(media_file_paths)

    if not image_units and not video_units:
        return {
            "status": "error",
            "message": "Could not plan any readable media.",
        }

    media_metadata = story_alias_metadata(
        build_story_media_metadata(media_file_paths)
    )

    processing = {
        "strategy": "map_reduce",
        "image_max_width": IMAGE_MAX_WIDTH,
        "video_frame_max_width": VIDEO_FRAME_MAX_WIDTH,
        "image_batch_size": IMAGE_BATCH_SIZE,
        "video_frames_per_video": VIDEO_FRAME_COUNT,
        "image_count": len(image_units),
        "video_count": len(video_units),
        "image_batches": 0,
        "videos_processed": 0,
    }

    try:
        if not needs_map_reduce(image_units, video_units):
            processing["strategy"] = "single_shot"

            story = clean_story_output(
                single_shot_images(
                    image_units=image_units,
                    media_metadata=media_metadata,
                    output_mode=output_mode,
                    recap_prompt=recap_prompt,
                    original_request=original_request,
                ),
                output_mode=output_mode,
            )

            return {
                "status": "success",
                "story": story,
                "output_mode": output_mode,
                "media_metadata": media_metadata,
                "processing": processing,
            }

        partials, map_stats = run_map_phase(
            image_units=image_units,
            video_units=video_units,
            recap_prompt=recap_prompt,
            output_mode=output_mode,
        )

        processing.update(map_stats)

        story = clean_story_output(
            reduce_phase(
                partials=partials,
                media_metadata=media_metadata,
                output_mode=output_mode,
                recap_prompt=recap_prompt,
                original_request=original_request,
            ),
            output_mode=output_mode,
        )

        return {
            "status": "success",
            "story": story,
            "output_mode": output_mode,
            "media_metadata": media_metadata,
            "processing": processing,
            "partials": [
                {
                    "label": partial["label"],
                    "kind": partial["kind"],
                }
                for partial in partials
            ],
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "output_mode": output_mode,
            "media_metadata": media_metadata,
            "processing": processing,
        }


def run_story_teller_from_uploads(
    uploaded_files: List[UploadedFile],
    recap_prompt: str = "",
    output_mode: str = "detailed_summary",
    original_request: str = "",
) -> Dict[str, Any]:
    temp_paths = []

    try:
        temp_paths = save_uploaded_media_files_to_temp(uploaded_files)

        return generate_story_teller_output(
            media_file_paths=temp_paths,
            recap_prompt=recap_prompt,
            output_mode=output_mode,
            original_request=original_request,
        )

    finally:
        remove_temp_files(temp_paths)


def ask_story_teller_follow_up(
    question: str,
    story_text: str,
    media_metadata: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    if not question or not question.strip():
        return {
            "status": "error",
            "message": "Question is empty.",
            "answer": "",
        }

    clean_media_metadata = story_alias_metadata(media_metadata or [])

    messages = [
        {
            "role": "system",
            "content": FOLLOWUP_SYSTEM,
        },
        {
            "role": "user",
            "content": (
                f"Generated story:\n{(story_text or '').strip()}\n\n"
                f"Media metadata:\n{json.dumps(clean_media_metadata, ensure_ascii=False)}"
            ),
        },
    ]

    for item in (chat_history or [])[-10:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()

        if role in {"user", "assistant"} and content:
            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    messages.append(
        {
            "role": "user",
            "content": question.strip(),
        }
    )

    try:
        answer = story_teller_chat_completion(
            messages,
            max_tokens=700,
            model=FOLLOWUP_MODEL,
        )
        answer = strip_model_reasoning(answer)

        # Reject / retry if the model still leaked planning text
        if not answer.strip() or looks_like_reasoning_dump(answer):
            messages.append(
                {
                    "role": "assistant",
                    "content": "(invalid internal draft discarded)",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Reply again with ONLY the user-facing text. "
                        "No analysis. No numbered steps. No 'Final answer:' label."
                    ),
                }
            )
            answer = strip_model_reasoning(
                story_teller_chat_completion(
                    messages,
                    max_tokens=500,
                    model=FOLLOWUP_MODEL,
                )
            )

        if looks_like_reasoning_dump(answer):
            answer = ""

        return {
            "status": "success",
            "answer": answer,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "answer": "",
        }