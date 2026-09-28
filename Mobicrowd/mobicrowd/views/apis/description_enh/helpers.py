"""Standalone text-enhancement API logic (no Text_enh / utils imports)."""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests
from django.conf import settings
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

# ----- config.py -----



OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Mobicrowd Description Enhancement")
DEFAULT_LLM_PROVIDER = "openrouter"


def _configured_openrouter_model() -> str:
    """Primary description-enhancement model from Django settings only."""
    model = str(
        getattr(settings, "OPENROUTER_DESCRIPTION_ENHANCEMENT_MODEL", "") or ""
    ).strip()
    if not model:
        raise RuntimeError(
            "OPENROUTER_DESCRIPTION_ENHANCEMENT_MODEL is not configured in Django settings."
        )
    return model


def _configured_openrouter_fallback_models() -> List[str]:
    """Ordered model fallback chain from Django settings only."""
    primary = _configured_openrouter_model()
    raw = getattr(
        settings,
        "OPENROUTER_DESCRIPTION_ENHANCEMENT_FALLBACK_MODELS",
        [],
    ) or []
    if isinstance(raw, str):
        raw = [raw]

    result: List[str] = []
    for value in raw:
        model = str(value or "").strip()
        if model and model != primary and model not in result:
            result.append(model)
    return result


def _configured_openrouter_keys_file() -> Path:
    """Centralized paid-key file configured by settings.OPENROUTER_KEYS_FILE."""
    raw = str(getattr(settings, "OPENROUTER_KEYS_FILE", "") or "").strip()
    if not raw:
        raise RuntimeError("OPENROUTER_KEYS_FILE is not configured in Django settings.")

    path = Path(raw).expanduser()
    if not path.is_file():
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {path}")
    return path
PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
}
SAFETY_GATE_VERSION = "utils_textual_safety"
REQUEST_KIND_CHECK_VERSION = "visual_textual_need_check_llm_v2"
REQUEST_KIND_VISUAL = "visual"
REQUEST_KIND_TEXTUAL = "textual"
RequestKind = Literal["visual", "textual"]
MIN_MEANINGFUL_WORDS = 3
UNSAFE_FOR_WORK_RECOMMENDED_MESSAGE = (
    "Submission note contains unsafe content. Please rewrite it."
)

LABELS: List[str] = [
    "target_object",
    "location_details",
    "media_type",
    "media_quantity",
    "action_required",
    "reward",
    "worker_requirements",
    "camera_specs",
    "do_and_donts",
    "time_details",
    "quantity_per_contributor",
    "safety_privacy",
]

# Single source of truth per label: summary (UI), extract (LLM prompt), question (follow-ups).
FIELD_SPECS: Dict[str, Dict[str, str]] = {
    "target_object": {
        "summary": "Main object, person, or thing(s) the task is about.",
        "extract": (
            "Subject of the visual media request: what must appear in, be shown by, or be the focus of "
            "the requested image/photo/video/screenshot (person, object, scene, category). "
            "For phrases like 'send images of cats' or 'provide photos showing cars', extract 'cats' "
            "or 'cars'. Copy a short noun phrase from the text. Not the action verb "
            "(-> action_required) and not the media format alone (-> media_type). "
            "If the text only uses meta placeholders such as 'main subject', 'subject', 'object', "
            "'thing', or 'something' without naming the actual subject, return null."
        ),
        "question": "What exactly should contributors show in the visual media - a person, object, or scene?",
    },
    "location_details": {
        "summary": "Place, address, city, venue, or geographic area.",
        "extract": (
            "Where the task happens or where content must be collected. "
            "Copy place names or location phrases (city, neighborhood, venue, address). null if no place is stated."
        ),
        "question": "Where should this task take place — a specific city, venue, or address?",
    },
    "media_type": {
        "summary": "Required output format (photo, video, audio, etc.).",
        "extract": (
            "The kind of media file required: image/images, photo/photos, picture/pictures, "
            "shot/shots, video/videos, screenshot/screenshots, gif/gifs, footage, etc. "
            "Copy format words from the text. Not the action verb (-> action_required)."
        ),
        "question": "What visual format is required - images, photos, shots, videos, screenshots, or something else?",
    },
    "media_quantity": {
        "summary": "Total number of files or amount of content required overall.",
        "extract": (
            "How many files, clips, or units are needed in total for the whole task "
            "(e.g. 30 photos, 5 videos). Return only the numeric quantity, such as 30 or 5. "
            "null if only per-worker limits are given or the quantity is vague."
        ),
        "question": "How many files, clips, or other units are needed in total for this task, across all contributors?",
    },
    "action_required": {
        "summary": "Required action or behavior — what workers must do (not the object).",
        "extract": (
            "The task verb or short verb phrase telling contributors what to do — copy it from the text "
            "(usually the first imperative: Capture, Take, Send, Provide, Document, Submit, Upload, "
            "Record, Film, Photograph, etc.). "
            "Examples: 'Provide 10 photos' → Provide; 'Document street art as photos' → Document. "
            "Not the subject (→ target_object), not the format alone (→ media_type), not worker rules (→ worker_requirements). "
            "Use null only for noun-only listings with no instruction verb, e.g. '30 cat photos in Tunis' "
            "(no Capture/Provide/Send/Document)."
        ),
        "question": (
            "What should contributors do, and what is the main action verb in the task "
            "(for example: send, provide, capture, document, upload, submit)?"
        ),
    },
    "reward": {
        "summary": "Payment or compensation for contributors.",
        "extract": (
            "How much workers are paid and on what basis ($ per photo, per hour, flat fee, etc.). "
            "Copy the payment phrase. null if compensation is not mentioned."
        ),
        "question": "What is the payment or compensation, and is it paid per item, per hour, or as a flat fee?",
    },
    "worker_requirements": {
        "summary": "Eligibility, skills, experience, rating, language, or device requirements for workers.",
        "extract": (
            "Who may do the task or what qualifications they need: experience, rating, age, language, "
            "owned equipment, certifications. Copy requirement sentences or phrases. "
            "Not the task action (→ action_required)."
        ),
        "question": "Are there eligibility requirements for contributors, such as experience, skill level, rating, age, language, or device ownership?",
    },
    "camera_specs": {
        "summary": "Camera or capture-device technical requirements.",
        "extract": (
            "Technical specs for capture gear: resolution, lens, smartphone model, tripod, etc. "
            "null if no device or camera constraints are stated."
        ),
        "question": "Are there camera, phone, or device specifications contributors must meet, such as resolution, model, lens, or tripod use?",
    },
    "do_and_donts": {
        "summary": "Rules, restrictions, or quality constraints for the work.",
        "extract": (
            "Explicit do's, don'ts, quality bars, framing rules, or submission constraints. "
            "Copy rule phrases. null if none are stated."
        ),
        "question": "Are there rules, restrictions, quality standards, privacy constraints, or submission requirements to mention?",
    },
    "time_details": {
        "summary": "When the task or capture must happen.",
        "extract": (
            "Dates, times, deadlines, time windows, or scheduling constraints "
            "(e.g. at midnight, before Friday, during rush hour). null if timing is open."
        ),
        "question": "When must the task be completed, such as a date, time window, deadline, or specific time of day?",
    },
    "quantity_per_contributor": {
        "summary": "How many submissions each contributor may provide.",
        "extract": (
            "Per-worker submission cap (e.g. max 3 per person). Return only the numeric quantity. "
            "null if only a total task quantity is given (→ media_quantity) or no per-person limit."
        ),
        "question": "How many submissions may each contributor make, and is there a per-person limit?",
    },
    "safety_privacy": {
        "summary": "Privacy, consent, safety, or anonymity requirements.",
        "extract": (
            "Consent, anonymity, blurring faces, no minors, license plates, private property, etc. "
            "null if no privacy or safety constraints are stated."
        ),
        "question": "Are there privacy, consent, anonymity, or safety requirements, such as avoiding faces, people, license plates, or private property details?",
    },
}

TEXTUAL_FIELD_OVERRIDES: Dict[str, Dict[str, str]] = {
    "target_object": {
        "summary": "Text content, document subject, question, or topic the request is about.",
        "extract": (
            "The content, topic, document subject, or question the requester wants handled in text. "
            "Copy a short noun phrase or question topic from the input. Not the action verb "
            "(-> action_required) and not only the output format (-> media_type). "
            "If the text only refers to 'the topic', 'the question', 'the content', or similar "
            "placeholders without stating the actual topic, return null."
        ),
        "question": "What content, document topic, or question should the text address?",
    },
    "media_type": {
        "summary": "Requested textual output format (answer, document, report, paragraph, etc.).",
        "extract": (
            "The kind of textual deliverable required: answer, document, report, essay, article, "
            "paragraph, point of view, review, email, caption, etc. Copy format words from the text. "
            "null if no textual format is stated."
        ),
        "question": "What textual format is required - answer, document, report, paragraph, or something else?",
    },
    "media_quantity": {
        "summary": "Total amount of text or number of textual deliverables required.",
        "extract": (
            "How much text or how many textual units are needed overall, such as word count, page count, "
            "number of answers, number of documents, or number of bullet points. Return only the numeric "
            "quantity. null if no amount is stated or the quantity is vague."
        ),
        "question": "How much text is needed overall - words, pages, answers, documents, or points?",
    },
    "action_required": {
        "summary": "Required textual action - what the requester asks for.",
        "extract": (
            "The task verb or short verb phrase telling what the requester wants done with the text/topic. "
            "Examples: write, answer, explain, summarize, review, translate, give a point of view, draft. "
            "Copy the action from the input. Not the topic (-> target_object) and not only the format "
            "(-> media_type)."
        ),
        "question": "What should be done with the topic - write, answer, explain, summarize, review, or something else?",
    },
    "camera_specs": {
        "summary": "Formatting, file, or technical requirements for the text/document.",
        "extract": (
            "Technical or formatting requirements for the textual deliverable: PDF, DOCX, Markdown, "
            "language, tone, citation format, file format, layout, or template. null if none are stated."
        ),
        "question": "Are there formatting, file, language, tone, or citation requirements for the text?",
    },
    "quantity_per_contributor": {
        "summary": "How many textual submissions each contributor may provide.",
        "extract": (
            "Per-worker text submission cap, such as max answers per person or one document per contributor. "
            "Return only the numeric quantity. "
            "null if only a total quantity is given (-> media_quantity) or no per-person limit."
        ),
        "question": "How many text submissions is each contributor allowed to provide?",
    },
}

TEXTUAL_FIELD_SPECS: Dict[str, Dict[str, str]] = {
    **FIELD_SPECS,
    **TEXTUAL_FIELD_OVERRIDES,
}

FIELD_SPECS_BY_REQUEST_KIND: Dict[str, Dict[str, Dict[str, str]]] = {
    REQUEST_KIND_VISUAL: FIELD_SPECS,
    REQUEST_KIND_TEXTUAL: TEXTUAL_FIELD_SPECS,
}

LABEL_DESCRIPTIONS: Dict[str, str] = {
    name: FIELD_SPECS[name]["summary"] for name in LABELS
}
LABEL_QUESTIONS: Dict[str, str] = {
    name: FIELD_SPECS[name]["question"] for name in LABELS
}
TEXTUAL_LABEL_QUESTIONS: Dict[str, str] = {
    name: TEXTUAL_FIELD_SPECS[name]["question"] for name in LABELS
}
LABEL_QUESTIONS_BY_REQUEST_KIND: Dict[str, Dict[str, str]] = {
    REQUEST_KIND_VISUAL: LABEL_QUESTIONS,
    REQUEST_KIND_TEXTUAL: TEXTUAL_LABEL_QUESTIONS,
}

PRIORITY_ORDER = [
    "target_object",
    "location_details",
    "media_type",
    "media_quantity",
    "time_details",
    "reward",
    "do_and_donts",
    "safety_privacy",
    "quantity_per_contributor",
    "worker_requirements",
    "camera_specs",
    "action_required",
]

TEXTUAL_PRIORITY_ORDER = [
    "target_object",
    "action_required",
    "media_type",
    "media_quantity",
    "reward",
    "time_details",
    "do_and_donts",
    "safety_privacy",
    "location_details",
    "quantity_per_contributor",
    "worker_requirements",
    "camera_specs",
]

PRIORITY_ORDER_BY_REQUEST_KIND: Dict[str, List[str]] = {
    REQUEST_KIND_VISUAL: PRIORITY_ORDER,
    REQUEST_KIND_TEXTUAL: TEXTUAL_PRIORITY_ORDER,
}

LLM_LABELS = LABELS.copy()

WORK_READY_REQUIRED_LABELS = ["target_object", "action_required"]
WORK_READY_REQUIRED_LABELS_BY_REQUEST_KIND: Dict[str, List[str]] = {
    REQUEST_KIND_VISUAL: WORK_READY_REQUIRED_LABELS,
    REQUEST_KIND_TEXTUAL: WORK_READY_REQUIRED_LABELS,
}

TARGET_OBJECT_QUALITY_VERSION = "target_object_quality_v1"

VAGUE_TARGET_EXACT = frozenset({
    "main subject",
    "the main subject",
    "subject",
    "the subject",
    "main object",
    "the main object",
    "object",
    "the object",
    "main target",
    "the main target",
    "target",
    "the target",
    "thing",
    "something",
    "anything",
    "item",
    "an item",
    "content",
    "the content",
    "scene",
    "the scene",
    "topic",
    "the topic",
    "main topic",
    "the main topic",
    "photo subject",
    "video subject",
    "image subject",
    "main focus",
    "the main focus",
    "focus",
    "the focus",
    "automated event",
    "event",
    "the event",
    "submission",
    "the submission",
    "deliverable",
    "the deliverable",
    "material",
    "the material",
    "the question",
    "the answer",
    "the document",
    "the text",
    "the report",
})

TARGET_OBJECT_META_TERMS = frozenset({
    "main",
    "subject",
    "object",
    "thing",
    "item",
    "content",
    "target",
    "scene",
    "topic",
    "something",
    "anything",
    "focus",
    "photo",
    "photos",
    "image",
    "images",
    "video",
    "videos",
    "picture",
    "pictures",
    "shot",
    "shots",
    "clip",
    "clips",
    "footage",
    "media",
    "visual",
    "clearly",
    "single",
    "capture",
    "submit",
    "deadline",
    "answer",
    "document",
    "text",
    "report",
    "question",
    "deliverable",
    "submission",
    "material",
    "entity",
    "element",
    "component",
    "aspect",
    "detail",
    "example",
    "sample",
})

SOFT_VAGUE_TARGET_TERMS = frozenset({
    "event",
    "activity",
    "product",
    "stuff",
    "data",
    "work",
    "project",
    "task",
    "unit",
    "material",
    "submission",
    "deliverable",
    "entity",
    "element",
    "component",
    "aspect",
    "detail",
    "example",
    "sample",
})

OPTIONAL_RECOMMENDED_LABELS = [
    "media_type",
    "media_quantity",
    "reward",
    "location_details",
    "time_details",
]

TEXTUAL_OPTIONAL_RECOMMENDED_LABELS = [
    "media_type",
    "media_quantity",
    "reward",
    "time_details",
    "do_and_donts",
    "camera_specs",
]

OPTIONAL_RECOMMENDED_LABELS_BY_REQUEST_KIND: Dict[str, List[str]] = {
    REQUEST_KIND_VISUAL: OPTIONAL_RECOMMENDED_LABELS,
    REQUEST_KIND_TEXTUAL: TEXTUAL_OPTIONAL_RECOMMENDED_LABELS,
}

STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "and", "or", "but", "if", "then", "than", "when", "where", "while",
    "whether", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that", "these",
    "those", "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "may", "might",
    "must", "can", "shall", "very", "really", "quite", "too", "also", "just", "only",
    "even", "still", "yet", "there", "here", "some", "any", "all", "each", "every",
    "both", "few", "more", "most", "other", "such", "no", "not", "so", "well"
}

ARABIC_STOP_WORDS = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "هنا", "هناك", "هو", "هي", "هم", "هن", "نحن", "أنا", "انا", "انت", "أنت",
    "أنتم", "ان", "أن", "إن", "إنه", "انه", "كانت", "كان", "يكون", "تكون",
    "و", "أو", "او", "ثم", "لكن", "بل", "كل", "أي", "اي", "لا", "ما", "لم", "لن",
    "قد", "لقد", "حتى", "بشكل", "جدا", "جدًا", "غير", "كما", "إذا", "اذا", "لذلك",
    "الذي", "التي", "الذين", "اللذين", "اللتي", "أيضا", "أيضًا", "فقط",
    "يجب", "يرجى", "رجاء", "مطلوب", "ضرورة", "بها", "به", "لها", "له", "لهم",
    "عند", "عبر", "ضمن", "قبل", "بعد", "بين", "خلال", "حول", "أمام", "امام",
}
ARABIC_STOP_WORDS |= {
    word.replace("أ", "ا")
    .replace("إ", "ا")
    .replace("آ", "ا")
    .replace("ى", "ي")
    .replace("ة", "ه")
    for word in ARABIC_STOP_WORDS
}

FEW_SHOT_EXAMPLES: List[Dict[str, object]] = [
    {
        "description": (
            "Capture 30 photos of people crossing streets in downtown Tunis. "
            "Reward is $2 per approved photo. At midnight."
        ),
        "expected": {
            "target_object": "people",
            "location_details": "downtown Tunis",
            "media_type": "photos",
            "media_quantity": "30",
            "action_required": "Capture",
            "reward": "$2 per approved photo",
            "worker_requirements": None,
            "camera_specs": None,
            "do_and_donts": None,
            "time_details": "At midnight",
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": (
            "Document street art as high-resolution photos in Melbourne. Workers must have "
            "photography experience of at least one year and a rating of at least 4.2."
        ),
        "expected": {
            "target_object": "street art",
            "location_details": "Melbourne",
            "media_type": "high-resolution photos",
            "media_quantity": None,
            "action_required": "Document",
            "reward": None,
            "worker_requirements": (
                "photography experience of at least one year and a rating of at least 4.2"
            ),
            "camera_specs": None,
            "do_and_donts": None,
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": (
            "Provide 10 photos. Workers must meet the following requirements: they must have "
            "photography experience of at least one year, a worker rating of at least 4.2, "
            "and expertise in urban photography."
        ),
        "expected": {
            "target_object": None,
            "location_details": None,
            "media_type": "photos",
            "media_quantity": "10",
            "action_required": "Provide",
            "reward": None,
            "worker_requirements": (
                "photography experience of at least one year, a worker rating of at least 4.2, "
                "and expertise in urban photography"
            ),
            "camera_specs": None,
            "do_and_donts": None,
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": "30 cat photos in Tunis. Pay $2 each.",
        "expected": {
            "target_object": "cat",
            "location_details": "Tunis",
            "media_type": "photos",
            "media_quantity": "30",
            "action_required": None,
            "reward": "$2 each",
            "worker_requirements": None,
            "camera_specs": None,
            "do_and_donts": None,
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": "Please send some images of cats.",
        "expected": {
            "target_object": "cats",
            "location_details": None,
            "media_type": "images",
            "media_quantity": None,
            "action_required": "send",
            "reward": None,
            "worker_requirements": None,
            "camera_specs": None,
            "do_and_donts": None,
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": (
            "Requester suite automated event. Please capture the main subject clearly in a single photo "
            "and submit before the deadline."
        ),
        "expected": {
            "target_object": None,
            "location_details": None,
            "media_type": "photo",
            "media_quantity": "1",
            "action_required": "Capture",
            "reward": None,
            "worker_requirements": None,
            "camera_specs": None,
            "do_and_donts": None,
            "time_details": "before the deadline",
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
]

TEXTUAL_FEW_SHOT_EXAMPLES: List[Dict[str, object]] = [
    {
        "description": (
            "Write a one-page document about recycling in schools. "
            "Use a formal tone and pay $15 per approved document."
        ),
        "expected": {
            "target_object": "recycling in schools",
            "location_details": None,
            "media_type": "one-page document",
            "media_quantity": "1",
            "action_required": "Write",
            "reward": "$15 per approved document",
            "worker_requirements": None,
            "camera_specs": "formal tone",
            "do_and_donts": None,
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": (
            "Answer the question: why should cities invest in public transport? "
            "Give a balanced point of view in 5 bullet points."
        ),
        "expected": {
            "target_object": "why cities should invest in public transport",
            "location_details": None,
            "media_type": "answer",
            "media_quantity": "5",
            "action_required": "Answer",
            "reward": None,
            "worker_requirements": None,
            "camera_specs": None,
            "do_and_donts": "Give a balanced point of view",
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
    {
        "description": (
            "Summarize this article about renewable energy into a short paragraph. "
            "Submit only original wording."
        ),
        "expected": {
            "target_object": "article about renewable energy",
            "location_details": None,
            "media_type": "short paragraph",
            "media_quantity": None,
            "action_required": "Summarize",
            "reward": None,
            "worker_requirements": None,
            "camera_specs": None,
            "do_and_donts": "Submit only original wording",
            "time_details": None,
            "quantity_per_contributor": None,
            "safety_privacy": None,
        },
    },
]

FEW_SHOT_EXAMPLES_BY_REQUEST_KIND: Dict[str, List[Dict[str, object]]] = {
    REQUEST_KIND_VISUAL: FEW_SHOT_EXAMPLES,
    REQUEST_KIND_TEXTUAL: TEXTUAL_FEW_SHOT_EXAMPLES,
}

PEOPLE_TERMS = re.compile(
    r"\b(person|people|human|humans|man|woman|men|women|child|children|kid|kids|minor|minors|"
    r"face|faces|pedestrian|pedestrians|customer|customers|worker|workers|student|students|"
    r"employee|employees|crowd|crowds|family|mother|mom|mum|mam|father|dad|sister|brother|"
    r"wife|husband|girlfriend|boyfriend)\b",
    re.IGNORECASE,
)

SAFE_PRIVACY_TERMS = re.compile(
    r"\b(avoid|do not|don't|without|blur|hide|mask|anonymous|anonymized|non-identifiable|"
    r"non identifiable|no faces|no face|consent|permission|public place|privacy|private information|"
    r"personal documents|license plates|plate numbers|no identifiable people|not identifiable)\b",
    re.IGNORECASE,
)

IGNORED_SAFETY_CATEGORIES = frozenset({
    "not_a_crowdsourcing_task",
    "privacy_invasive_request",
    "privacy_sensitive_without_guardrail",
})

ALLOWED_SAFETY_CATEGORIES = frozenset({
    "impolite_or_unprofessional_language",
    "privacy_sensitive_without_guardrail",
    "sexual_or_adult_content",
    "violence_or_harm",
    "hate_or_discrimination",
    "illegal_or_dangerous",
    "empty_or_meaningless_request",
    "safety_check_unavailable",
})

VISUAL_MEDIA_TERMS = re.compile(
    r"\b(photo|photos|image|images|picture|pictures|video|videos|clip|clips|footage|"
    r"shot|shots|screenshot|screenshots|gif|gifs|camera|cameras|photograph|photographs)\b",
    re.IGNORECASE,
)

VISUAL_ACTION_TERMS = re.compile(
    r"\b(capture|take|record|film|shoot|photograph|send|upload|submit|provide|collect|document)\b",
    re.IGNORECASE,
)

TEXTUAL_OUTPUT_TERMS = re.compile(
    r"\b(text|document|documents|doc|docs|answer|answers|question|questions|qst|article|"
    r"essay|report|summary|paragraph|bullet points|points|point of view|pov|opinion|review|"
    r"explanation|email|letter|blog|post|caption|captions|copy|script|prompt|prompts)\b",
    re.IGNORECASE,
)

TEXTUAL_ACTION_TERMS = re.compile(
    r"\b(write|draft|compose|answer|explain|summarize|summarise|rewrite|edit|proofread|"
    r"translate|review|give|create|generate|describe|list|compare|argue|analyze|analyse|"
    r"think|believe|discuss|respond)\b",
    re.IGNORECASE,
)

TEXTUAL_QUESTION_OR_OPINION_TERMS = re.compile(
    r"(\?|؟)|\b(what|why|how|when|where|who|which|think|thought|thoughts|opinion|"
    r"view|perspective|believe|feel|question|qst|answer|response)\b|"
    r"(ما|ماذا|لماذا|كيف|متى|أين|اين|رأي|راي|وجهة نظر|تعتقد|اشرح|اكتب|أجب|اجب)",
    re.IGNORECASE,
)

STRONG_VISUAL_REQUEST = re.compile(
    r"\b(capture|take|record|film|shoot|photograph|send|upload|submit|provide|collect|document)\b"
    r"[\s\S]{0,50}\b(photo|photos|image|images|picture|pictures|video|videos|clip|clips|footage|"
    r"shot|shots|screenshot|screenshots|gif|gifs)\b",
    re.IGNORECASE,
)

STRONG_TEXTUAL_REQUEST = re.compile(
    r"\b(write|draft|compose|answer|explain|summarize|summarise|rewrite|edit|proofread|"
    r"translate|review|give|create|generate|describe|list|compare|argue|analyze|analyse)\b"
    r"[\s\S]{0,70}\b(text|document|documents|doc|docs|answer|answers|question|questions|qst|"
    r"article|essay|report|summary|paragraph|bullet points|points|point of view|pov|opinion|"
    r"review|explanation|email|letter|blog|post|caption|captions|copy|script|prompt|prompts)\b",
    re.IGNORECASE,
)

VISUAL_SELF_EVIDENT_ACTIONS = frozenset({
    "capture",
    "take",
    "record",
    "film",
    "shoot",
    "photograph",
})

# ----- paths.py -----



_API_ENHANCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_API_ENHANCE_DIR)

TITLE_GEN_DIR = os.path.join(_API_ENHANCE_DIR, "Title_gen")

# ----- helpers.py -----





def dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in items if x]))


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return s[int(k)] if f == c else s[f] * (c - k) + s[c] * (k - f)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def meaningful_words(text: str) -> List[str]:
    """
    Return meaningful words from English, Arabic, and mixed descriptions.
    """
    text = (text or "").lower()
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    words = re.findall(
        r"[a-zA-Z]+|[\u0600-\u06FF]+|\d+",
        text,
        flags=re.UNICODE,
    )

    meaningful: List[str] = []
    for word in words:
        word = word.strip()
        if len(word) <= 1:
            continue
        if word in STOP_WORDS:
            continue
        if word in ARABIC_STOP_WORDS:
            continue
        meaningful.append(word)
    return meaningful


def _normalize_request_kind(request_kind: Optional[str]) -> RequestKind:
    kind = (request_kind or REQUEST_KIND_VISUAL).strip().lower()
    if kind == REQUEST_KIND_TEXTUAL:
        return REQUEST_KIND_TEXTUAL
    return REQUEST_KIND_VISUAL


def _questions_for_request_kind(request_kind: Optional[str]) -> Dict[str, str]:
    return LABEL_QUESTIONS_BY_REQUEST_KIND[_normalize_request_kind(request_kind)]


def _field_specs_for_request_kind(request_kind: Optional[str]) -> Dict[str, Dict[str, str]]:
    return FIELD_SPECS_BY_REQUEST_KIND[_normalize_request_kind(request_kind)]


def _priority_order_for_request_kind(request_kind: Optional[str]) -> List[str]:
    return PRIORITY_ORDER_BY_REQUEST_KIND[_normalize_request_kind(request_kind)]


def _required_labels_for_request_kind(request_kind: Optional[str]) -> List[str]:
    return WORK_READY_REQUIRED_LABELS_BY_REQUEST_KIND[_normalize_request_kind(request_kind)]


def _optional_labels_for_request_kind(request_kind: Optional[str]) -> List[str]:
    return OPTIONAL_RECOMMENDED_LABELS_BY_REQUEST_KIND[_normalize_request_kind(request_kind)]


def top_missing_labels(
    extractions: Dict[str, Optional[str]],
    k: int = 5,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> List[str]:
    priority_order = _priority_order_for_request_kind(request_kind)
    return [lbl for lbl in priority_order if not extractions.get(lbl)][:k]


def _pattern_hits(pattern: re.Pattern, text: str) -> List[str]:
    return dedupe([m.group(0).strip().lower() for m in pattern.finditer(text or "")])


def _request_kind_prompt(description: str) -> Tuple[str, str]:
    system_prompt = """You classify a safe crowdsourcing request into exactly one processing type.

Return ONLY valid JSON with this schema:
{
  "request_type": "visual or textual",
  "confidence": 0.0,
  "reason": "short reason"
}

Definitions:
- visual: the requester needs visual media files as the deliverable, such as photos, images, pictures, shots, screenshots, videos, clips, footage, or camera capture. This includes asking contributors to send, provide, upload, submit, capture, take, record, film, or document visual media, whether the media is newly captured or already exists.
- textual: the requester needs text as the deliverable or asks a question/opinion task, such as an answer, document, report, paragraph, summary, explanation, review, point of view, or "what do you think" request.

Decision rules:
- Choose textual for questions and opinion prompts even if no explicit word "text" appears.
- Choose textual when images are only the input/context but the requested deliverable is text, e.g. captions or descriptions.
- Choose visual when the requested deliverable is visual media, e.g. "send images of cats", "provide photos of receipts", or "upload a screenshot".
- Do not require capture wording for visual; "send/provide/upload images/photos/videos of X" is visual.
- If ambiguous and there is no explicit visual-media deliverable, choose textual.
- Do not perform safety review here; safety already ran.

Examples:
- "what do you think about artificial intelligence" -> textual
- "give a point of view about remote work" -> textual
- "write captions for these images" -> textual
- "please send some images of cats" -> visual
- "provide photos of receipts" -> visual
- "upload a screenshot of the app" -> visual
- "capture 30 photos of cats" -> visual
- "document street art as high-resolution photos" -> visual
"""
    user_prompt = f"""Request:
{description.strip()}

Classify the processing type. Return JSON only."""
    return system_prompt, user_prompt


def _request_kind_payload(
    request_kind: str,
    *,
    method: str,
    confidence: float,
    reason: str,
    signals: Optional[Dict[str, Any]] = None,
    raw_model_output: str = "",
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    request_kind = _normalize_request_kind(request_kind)
    payload = {
        "status": "TEXTUAL" if request_kind == REQUEST_KIND_TEXTUAL else "VISUAL",
        "request_type": request_kind,
        "prompt_profile": (
            "textual_content_extraction"
            if request_kind == REQUEST_KIND_TEXTUAL
            else "visual_media_extraction"
        ),
        "ran": True,
        "checked_by": (
            "request_kind_llm_classifier"
            if method.startswith("llm")
            else "request_kind_classifier"
        ),
        "method": method,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "reason": reason,
        "version": REQUEST_KIND_CHECK_VERSION,
        "signals": signals or {},
    }
    if raw_model_output:
        payload["raw_model_output"] = raw_model_output
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    return payload


def _keyword_request_kind_check(
    description: str,
    *,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fallback classifier used only when the prompt-based classifier is unavailable.

    Visual requests keep the existing media pipeline. Textual requests use the
    alternate prompt where target_object means content/topic and action_required
    means what the requester asks to do with that content.
    """
    text = clean_text(description)
    visual_media_hits = _pattern_hits(VISUAL_MEDIA_TERMS, text)
    visual_action_hits = _pattern_hits(VISUAL_ACTION_TERMS, text)
    textual_output_hits = _pattern_hits(TEXTUAL_OUTPUT_TERMS, text)
    textual_action_hits = _pattern_hits(TEXTUAL_ACTION_TERMS, text)
    textual_question_hits = _pattern_hits(TEXTUAL_QUESTION_OR_OPINION_TERMS, text)

    if visual_media_hits:
        visual_action_score = len(visual_action_hits)
    else:
        visual_action_score = len([
            hit for hit in visual_action_hits if hit in VISUAL_SELF_EVIDENT_ACTIONS
        ])

    visual_score = len(visual_media_hits) + min(visual_action_score, 2)
    textual_score = (
        len(textual_output_hits)
        + min(len(textual_action_hits), 2)
        + min(len(textual_question_hits), 2)
    )

    strong_visual = bool(STRONG_VISUAL_REQUEST.search(text))
    strong_textual = bool(STRONG_TEXTUAL_REQUEST.search(text))
    if strong_visual:
        visual_score += 3
    if strong_textual:
        textual_score += 3

    # Requests like "write captions for these images" mention images, but the
    # deliverable is text, so let a strong textual action win.
    if strong_textual and not strong_visual:
        request_kind = REQUEST_KIND_TEXTUAL
    elif strong_visual and visual_score >= textual_score:
        request_kind = REQUEST_KIND_VISUAL
    elif visual_score == 0 and textual_score > 0:
        request_kind = REQUEST_KIND_TEXTUAL
    elif textual_score > visual_score:
        request_kind = REQUEST_KIND_TEXTUAL
    elif visual_score == 0:
        request_kind = REQUEST_KIND_TEXTUAL
    else:
        request_kind = REQUEST_KIND_VISUAL

    total = max(1, visual_score + textual_score)
    winning_score = textual_score if request_kind == REQUEST_KIND_TEXTUAL else visual_score
    confidence = round(max(0.51, min(0.99, winning_score / total)), 2)

    reasons = {
        "visual_media_terms": visual_media_hits,
        "visual_action_terms": visual_action_hits,
        "textual_output_terms": textual_output_hits,
        "textual_action_terms": textual_action_hits,
        "textual_question_or_opinion_terms": textual_question_hits,
        "strong_visual_match": strong_visual,
        "strong_textual_match": strong_textual,
    }

    return _request_kind_payload(
        request_kind,
        method=(
            "keyword_profile_rules_fallback"
            if fallback_reason
            else "keyword_profile_rules"
        ),
        confidence=confidence,
        reason=(
            "Textual request detected; using textual content extraction prompt."
            if request_kind == REQUEST_KIND_TEXTUAL
            else "Visual media request detected or defaulted; using existing visual media extraction prompt."
        ),
        signals=reasons,
        fallback_reason=fallback_reason,
    )


def detect_request_kind(
    description: str,
    backend: Optional["OpenRouterBackend"] = None,
) -> Dict[str, Any]:
    """
    Prompt-based request-type check. Falls back to local signals only if the
    LLM classifier is unavailable or returns an invalid payload.
    """
    description = clean_text(description)
    if backend is None:
        return _keyword_request_kind_check(
            description,
            fallback_reason="No backend was provided for LLM request-type classification.",
        )

    system_prompt, user_prompt = _request_kind_prompt(description)
    try:
        raw = backend.chat(
            system_prompt=system_prompt,
            user_message=user_prompt,
            max_tokens=180,
        )
        obj = try_parse_json(raw)
        if not isinstance(obj, dict):
            raise ValueError("LLM request-type classifier returned non-JSON output.")

        raw_kind = str(obj.get("request_type", "")).strip().lower()
        if raw_kind not in {REQUEST_KIND_VISUAL, REQUEST_KIND_TEXTUAL}:
            if "text" in raw_kind or "opinion" in raw_kind or "question" in raw_kind:
                raw_kind = REQUEST_KIND_TEXTUAL
            elif "visual" in raw_kind or "media" in raw_kind or "photo" in raw_kind:
                raw_kind = REQUEST_KIND_VISUAL
        if raw_kind not in {REQUEST_KIND_VISUAL, REQUEST_KIND_TEXTUAL}:
            raise ValueError(f"Invalid request_type from LLM: {raw_kind or '<empty>'}.")

        try:
            confidence = float(obj.get("confidence", 0.85))
        except Exception:
            confidence = 0.85

        return _request_kind_payload(
            raw_kind,
            method="llm_prompt_classifier",
            confidence=confidence,
            reason=str(obj.get("reason") or "").strip()
            or "LLM classified the request type.",
            raw_model_output=raw,
        )
    except Exception as exc:
        return _keyword_request_kind_check(
            description,
            fallback_reason=f"LLM request-type classification failed: {exc}",
        )


def _skipped_request_kind_check() -> Dict[str, Any]:
    return {
        "status": "SKIPPED_UNSAFE",
        "request_type": None,
        "prompt_profile": None,
        "ran": False,
        "checked_by": "request_kind_classifier",
        "method": "keyword_profile_rules",
        "confidence": 0.0,
        "reason": "Skipped because safety check did not pass.",
        "version": REQUEST_KIND_CHECK_VERSION,
        "signals": {},
    }


def try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?|```", "", text.strip()).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)

    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

# ----- openrouter_backend.py -----

_KEY_LOCK = Lock()
_KEY_INDEX = 0


def _parse_api_keys(raw: str) -> List[str]:
    """
    Supports:
    - one key per line
    - comma-separated keys
    - optional comments using #
    - optional 'Bearer ' prefix
    """
    if not raw:
        return []

    raw = raw.replace(",", "\n")
    keys: List[str] = []

    for line in raw.splitlines():
        key = line.strip()

        if not key:
            continue

        if key.startswith("#"):
            continue

        if key.lower().startswith("bearer "):
            key = key[7:].strip()

        if "=" in key:
            key = key.split("=", 1)[1].strip()

        if key:
            keys.append(key)

    return keys


def _dedupe_keys(keys: List[str]) -> List[str]:
    seen = set()
    clean_keys = []

    for key in keys:
        if key not in seen:
            clean_keys.append(key)
            seen.add(key)

    return clean_keys


def get_openrouter_api_keys() -> List[str]:
    """
    Load OpenRouter credentials only from settings.OPENROUTER_KEYS_FILE.

    No request-level key, helper-local key file, or environment-key fallback is
    accepted here. This keeps the paid OpenRouter credential centralized.
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
    """
    Rotate key only for key/quota-related API failures.

    401/403: invalid or unauthorized key
    402: payment / credits required (OpenRouter)
    429: rate limit
    """
    status_code = getattr(exc, "status_code", None)

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", status_code)

    return status_code in {401, 402, 403, 429}


def provider_used_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def build_extraction_field_guide(
    labels: List[str],
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> str:
    """Build LLM field guide from FIELD_SPECS (no per-field prompt hardcoding)."""
    blocks = []
    field_specs = _field_specs_for_request_kind(request_kind)
    for name in labels:
        spec = field_specs[name]
        blocks.append(f"- {name}: {spec['extract']}")
    return "\n".join(blocks)


def build_few_shot_examples_block(
    labels: List[str],
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> str:
    """Render all few-shot pairs, keeping only keys requested for this extraction."""
    parts = []
    examples = FEW_SHOT_EXAMPLES_BY_REQUEST_KIND[_normalize_request_kind(request_kind)]
    for fewshot in examples:
        expected = fewshot["expected"]
        if not isinstance(expected, dict):
            continue
        fewshot_json = json.dumps(
            {k: expected[k] for k in labels if k in expected},
            ensure_ascii=False,
        )
        parts.append(f"Input: {fewshot['description']}\nOutput: {fewshot_json}")
    return "\n\n".join(parts)


def build_system_prompt(
    labels: Optional[List[str]] = None,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> str:
    labels = labels or LLM_LABELS
    request_kind = _normalize_request_kind(request_kind)
    field_guide = build_extraction_field_guide(labels, request_kind)
    examples_text = build_few_shot_examples_block(labels, request_kind)
    profile_note = ""
    if request_kind == REQUEST_KIND_TEXTUAL:
        profile_note = (
            "\nPrompt profile: textual\n"
            "This is a textual request. Treat target_object as the content/topic/question, "
            "and action_required as what the requester asks to do with that content.\n"
        )

    return f"""You extract structured fields from crowdsourcing task descriptions.
{profile_note}

Return ONLY a valid JSON object with EXACTLY these keys:
{json.dumps(labels, ensure_ascii=False)}

Global rules:
1) Copy short evidence spans from the input; do not invent details.
2) Use null for any field not stated or not clearly supported by the text.
3) Do not copy meta placeholders as target_object (e.g. main subject, object, thing, topic).
4) No markdown, no commentary, no extra keys.

Field extraction guide:
{field_guide}

Examples:
{examples_text}
"""


def build_user_message(description: str) -> str:
    return f'Task description:\n"{description}"\n\nReturn JSON only.'


POLISH_SYSTEM_PROMPT = (
    "You are a copy editor for crowdsourcing task briefs. "
    "You clean up the requester's own words: fix spelling, grammar and punctuation, "
    "drop repeated or filler wording, and keep every instruction the requester actually wrote. "
    "You never invent counts, media types, categories, examples, places, times or purposes. "
    "Write in the same language as the original request."
)

_POLISH_STRICT_REMINDER = (
    "\nSTRICT RETRY: your previous answer added information that is not in the original "
    "request. Rewrite again using ONLY the wording, counts and media types that appear in "
    "the original request. Add nothing.\n"
)


def build_polish_prompt(
    draft: str,
    source_description: str = "",
    strict: bool = False,
) -> str:
    reference_block = ""
    if draft.strip():
        reference_block = f"""
Extracted fields (reference only — do not add anything that is not in the original request):
{draft.strip()}
"""
    return f"""Clean up the original request below so crowd workers can read it easily.
{_POLISH_STRICT_REMINDER if strict else ""}
FIX:
- Spelling, grammar, capitalization and punctuation (e.g. "ure" -> "your", "captown" -> "Cape Town").
- Repeated words, duplicated phrases and filler wording.
- Make it a polite, direct instruction (e.g. "Please provide...", "Please capture...").

KEEP:
- Every item, count, media type, subject, place, time and condition the requester wrote.
- The requester's own intent and story, in the requester's language.

NEVER:
- Never add counts or quantities that are not in the original request.
- Never add media types (photos, videos, images, clips) that are not in the original request.
- Never add categories, examples, equipment lists, places, deadlines or quality rules.
- Never add a purpose such as "for data collection purposes", "for research" or "for the requested purpose".
- Never add qualifiers like "at least", "exactly" or "minimum".
- Never repeat the original text before your answer, and never add commentary.

Return ONE clean paragraph. No bullets. No markdown.

Original request (rewrite THIS):
\"{source_description.strip()}\"
{reference_block}
Cleaned description:
"""


class OpenRouterBackend:
    provider = DEFAULT_LLM_PROVIDER
    endpoint = OPENROUTER_ENDPOINT

    def __init__(
        self,
        temperature: float = 0.0,
    ):
        self.model_name = _configured_openrouter_model()
        self.fallback_models = _configured_openrouter_fallback_models()
        self.model_chain = [self.model_name, *self.fallback_models]
        self.temperature = temperature
        self.api_keys = get_openrouter_api_keys()
        self.system_prompt = build_system_prompt(LLM_LABELS)

    def chat(self, system_prompt: str, user_message: str, max_tokens: int = 512) -> str:
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
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                        "provider": {"allow_fallbacks": True},
                    },
                    timeout=(5.0, 60.0),
                )
                response.raise_for_status()
                payload = response.json()

                choices = payload.get("choices") or []
                if not choices:
                    return ""

                message = choices[0].get("message") or {}
                return str(message.get("content") or "").strip()

            except requests.HTTPError as exc:
                last_error = exc
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

    def predict(self, description: str) -> str:
        return self.chat(
            system_prompt=self.system_prompt,
            user_message=build_user_message(description),
            max_tokens=512,
        )

    def predict_for_request_kind(
        self,
        description: str,
        request_kind: Optional[str] = REQUEST_KIND_VISUAL,
    ) -> str:
        request_kind = _normalize_request_kind(request_kind)
        if request_kind == REQUEST_KIND_VISUAL:
            return self.predict(description)
        return self.chat(
            system_prompt=build_system_prompt(LLM_LABELS, request_kind=request_kind),
            user_message=build_user_message(description),
            max_tokens=512,
        )

    def polish_description(
        self,
        draft: str,
        source_description: str = "",
        strict: bool = False,
    ) -> str:
        out = self.chat(
            system_prompt=POLISH_SYSTEM_PROMPT,
            user_message=build_polish_prompt(draft, source_description, strict=strict),
            max_tokens=512,
        )
        return clean_text(out)

# ----- extraction.py -----


def normalize_extractions(parsed: Optional[dict]) -> Dict[str, Optional[str]]:
    parsed = parsed or {}
    out: Dict[str, Optional[str]] = {}
    null_markers = {"none", "null", "n/a", "absent", "not mentioned", "not specified"}

    for label in LABELS:
        value = parsed.get(label)

        if value is None:
            out[label] = None
        else:
            text = value.strip() if isinstance(value, str) else str(value).strip()
            if not text or text.lower() in null_markers:
                out[label] = None
                continue
            if label in {"media_quantity", "quantity_per_contributor"}:
                out[label] = _normalize_media_count_value(text)
            else:
                out[label] = text

    return out


def parse_llm_output(raw_text: str) -> Tuple[Dict[str, Optional[str]], bool]:
    obj = try_parse_json(raw_text)
    if obj is not None:
        return normalize_extractions(obj), True
    return {label: None for label in LABELS}, False


def extract_fields(
    description: str,
    backend: OpenRouterBackend,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Tuple[str, Dict[str, Optional[str]]]:
    raw_output = backend.predict_for_request_kind(description, request_kind)
    parsed = try_parse_json(raw_output)
    extractions = normalize_extractions(parsed)
    return raw_output, extractions


def extraction_summary(
    extractions: Dict[str, Optional[str]],
    raw: str,
    parsed_ok: bool,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Dict[str, object]:
    questions = _questions_for_request_kind(request_kind)
    missing = top_missing_labels(extractions, k=5, request_kind=request_kind)
    return {
        "ok": parsed_ok,
        "extractions": extractions,
        "raw_model_output": raw,
        "missing_labels": missing,
        "missing_questions": {label: questions[label] for label in missing},
    }


# ----- media autofill extraction -----

MEDIA_AUTOFILL_TYPES = ("image", "text", "video")
MEDIA_AUTOFILL_FIELDS = ("reward", "number_total", "number_per_contributor")


MEDIA_AUTOFILL_SYSTEM_PROMPT = """You extract media-specific autofill fields from crowdsourcing task descriptions.

Return ONLY a valid JSON object with exactly this shape:
{
  "image": {"reward": null, "number_total": null, "number_per_contributor": null},
  "text": {"reward": null, "number_total": null, "number_per_contributor": null},
  "video": {"reward": null, "number_total": null, "number_per_contributor": null}
}

Rules:
1) Fill a value only when it is explicitly stated in the input.
2) For reward, return only the numeric payment amount. Do not include currency symbols, words, or per-media text.
3) For number_total and number_per_contributor, return only the numeric quantity. Do not include media words like photos, shots, videos, reviews, or clips.
4) Use null when the value is absent, unclear, inferred, or belongs to another media type.
5) image includes photos, images, pictures, shots, screenshots, and gifs.
6) text includes written answers, reviews, reports, documents, paragraphs, captions, and essays.
7) video includes videos, clips, footage, recordings, and films.
8) If a value is global and only one media type is requested, put it under that media type.
9) If multiple media types are requested and a global value is not clearly tied to each type, leave the per-media value null.
10) Do not add extra keys, markdown, or commentary.

Examples:
Input: Capture 30 photos of storefronts. Pay $2 per approved photo. Max 5 photos per contributor.
Output: {"image":{"reward":"2","number_total":"30","number_per_contributor":"5"},"text":{"reward":null,"number_total":null,"number_per_contributor":null},"video":{"reward":null,"number_total":null,"number_per_contributor":null}}

Input: Write 12 short product reviews in one paragraph each. Reward is $1 per review.
Output: {"image":{"reward":null,"number_total":null,"number_per_contributor":null},"text":{"reward":"1","number_total":"12","number_per_contributor":null},"video":{"reward":null,"number_total":null,"number_per_contributor":null}}

Input: Submit 10 photos and 2 videos of the venue setup. Pay $1 per photo and $3 per video. Each contributor can submit up to 5 photos and 1 video.
Output: {"image":{"reward":"1","number_total":"10","number_per_contributor":"5"},"text":{"reward":null,"number_total":null,"number_per_contributor":null},"video":{"reward":"3","number_total":"2","number_per_contributor":"1"}}

Input: Contributors can also send some shots up to 5 shots. Each shot pays $0.2.
Output: {"image":{"reward":"0.2","number_total":null,"number_per_contributor":"5"},"text":{"reward":null,"number_total":null,"number_per_contributor":null},"video":{"reward":null,"number_total":null,"number_per_contributor":null}}
"""


def _empty_media_autofill() -> Dict[str, Dict[str, Optional[str]]]:
    return {
        media_type: {field: None for field in MEDIA_AUTOFILL_FIELDS}
        for media_type in MEDIA_AUTOFILL_TYPES
    }


MEDIA_COUNT_VALUE_RE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:,\d{3})*(?:\.\d+)?)?\b",
    re.IGNORECASE,
)
MEDIA_REWARD_VALUE_RE = re.compile(
    r"(?:[$]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:usd|dollars?)?",
    re.IGNORECASE,
)
MEDIA_COUNT_UNIT_RE = re.compile(
    r"\b("
    r"photos?|images?|pictures?|shots?|screenshots?|gifs?|videos?|clips?|footage|recordings?|films?|"
    r"answers?|reviews?|reports?|documents?|paragraphs?|captions?|essays?"
    r")\b",
    re.IGNORECASE,
)
NUMBER_WORD_VALUES = {
    "a": "1",
    "an": "1",
    "one": "1",
    "single": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
NUMBER_WORD_RE = re.compile(
    r"\b(a|an|one|single|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
    re.IGNORECASE,
)
MEDIA_COUNT_TOKEN_PATTERN = (
    r"(?:\d+(?:,\d{3})*(?:\.\d+)?|a|an|one|single|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)"
)
MEDIA_AUTOFILL_UNIT_PATTERNS = {
    "image": r"(?:photos?|images?|pictures?|shots?|screenshots?|gifs?)",
    "text": r"(?:answers?|reviews?|reports?|documents?|paragraphs?|captions?|essays?)",
    "video": r"(?:videos?|clips?|footage|recordings?|films?)",
}
MEDIA_PER_CONTRIBUTOR_CONTEXT_RE = re.compile(
    r"\b(each|every|per|contributor|contributors|person|persons|worker|workers|"
    r"participant|participants|up to|max|maximum|at most|no more than|allowed)\b",
    re.IGNORECASE,
)


def _normalize_media_count_value(text: str) -> Optional[str]:
    match = MEDIA_COUNT_VALUE_RE.search(text)
    if match:
        return clean_text(match.group(0))

    word_match = NUMBER_WORD_RE.search(text)
    if word_match and MEDIA_COUNT_UNIT_RE.search(text):
        return NUMBER_WORD_VALUES[word_match.group(1).lower()]

    return None


def _normalize_media_reward_value(text: str) -> Optional[str]:
    match = MEDIA_REWARD_VALUE_RE.search(text)
    if not match:
        return None
    return clean_text(match.group(1).replace(",", ""))


def _normalize_media_autofill(parsed: Optional[dict]) -> Dict[str, Dict[str, Optional[str]]]:
    normalized = _empty_media_autofill()
    if not isinstance(parsed, dict):
        return normalized

    null_markers = {"", "none", "null", "n/a", "absent", "not mentioned", "not specified"}
    for media_type in MEDIA_AUTOFILL_TYPES:
        values = parsed.get(media_type)
        if not isinstance(values, dict):
            continue
        for field in MEDIA_AUTOFILL_FIELDS:
            value = values.get(field)
            if value is None:
                continue
            text = clean_text(str(value))
            if text.lower() in null_markers:
                continue
            if field == "reward":
                normalized_reward = _normalize_media_reward_value(text)
                if not normalized_reward or normalized_reward.lower() in null_markers:
                    continue
                text = normalized_reward
            elif field in {"number_total", "number_per_contributor"}:
                normalized_count = _normalize_media_count_value(text)
                if not normalized_count or normalized_count.lower() in null_markers:
                    continue
                text = normalized_count
            normalized[media_type][field] = text
    return normalized


def _first_media_count_match(
    description: str,
    media_type: str,
    patterns: List[str],
    *,
    reject_per_contributor_context: bool = False,
) -> Optional[str]:
    units = MEDIA_AUTOFILL_UNIT_PATTERNS[media_type]
    for pattern in patterns:
        match = re.search(pattern.format(count=MEDIA_COUNT_TOKEN_PATTERN, units=units), description, re.IGNORECASE)
        if not match:
            continue
        if reject_per_contributor_context:
            context = description[max(0, match.start() - 60): match.end() + 25]
            if MEDIA_PER_CONTRIBUTOR_CONTEXT_RE.search(context):
                continue
        count = _normalize_media_count_value(match.group(0))
        if count:
            return count
    return None


def _apply_media_autofill_count_fallbacks(
    description: str,
    media_autofill: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Dict[str, Optional[str]]]:
    text = clean_text(description)
    if not text:
        return media_autofill

    per_contributor_patterns = [
        r"\b(?:up to|max(?:imum)?|at most|no more than)\s+{count}\s+{units}\b",
        r"\b{count}\s+{units}\s+(?:per|for each)\s+(?:contributor|person|worker|participant)\b",
        r"\b(?:each|every)\s+(?:contributor|person|worker|participant)[^.]{{0,80}}?\b{count}\s+{units}\b",
        r"\b(?:contributors?|workers?|participants?)\s+(?:can|may|are allowed to|allowed to)[^.]{{0,80}}?\b{count}\s+{units}\b",
    ]
    total_patterns = [
        r"\b(?:capture|record|submit|send|provide|need|needs|required|collect|take|upload)\s+{count}\s+{units}\b",
        r"\b(?:total|overall)\s+(?:of\s+)?{count}\s+{units}\b",
        r"\b{count}\s+{units}\s+(?:in total|overall|required|needed)\b",
    ]

    for media_type in MEDIA_AUTOFILL_TYPES:
        values = media_autofill[media_type]
        if not values.get("number_per_contributor"):
            values["number_per_contributor"] = _first_media_count_match(text, media_type, per_contributor_patterns)
        if not values.get("number_total"):
            values["number_total"] = _first_media_count_match(
                text,
                media_type,
                total_patterns,
                reject_per_contributor_context=True,
            )
    return media_autofill


def _build_media_autofill_user_prompt(description: str) -> str:
    return f"""Task description:
{description.strip()}

Return JSON only."""


def extract_media_autofill_fields(
    description: str,
    backend: OpenRouterBackend,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], str, bool]:
    raw = backend.chat(
        system_prompt=MEDIA_AUTOFILL_SYSTEM_PROMPT,
        user_message=_build_media_autofill_user_prompt(description),
        max_tokens=500,
    )
    parsed = try_parse_json(raw)
    media_autofill = _normalize_media_autofill(parsed)
    media_autofill = _apply_media_autofill_count_fallbacks(description, media_autofill)
    return media_autofill, raw, parsed is not None


# ----- SAM3 prompt enrichment -----

SAM3_RELATED_PROMPTS_FEWSHOT = """
You generate additional segmentation labels for SAM-style text prompts.

Rules:
- Input already contains TARGET_OBJECT items. Do NOT repeat them.
- Propose additional related, visually detectable objects likely to appear in the scene.
- Prefer concrete nouns / short noun phrases (1-3 words).
- Avoid abstract terms, verbs, and media words (photo/video/image).
- Respect RESTRICTIONS: do not output excluded terms.
- Output ONLY a comma-separated list (no bullets, no extra text).
- Return between 3 and 10 items.

Example 1
Task: Photograph wildfire containment efforts; capture fire lines, equipment, and personnel.
TARGET_OBJECT: fire lines, equipment, personnel
RESTRICTIONS: None
Output: smoke, firefighter, fire truck, hose, hard hat, flames, bulldozer

Example 2
Task: Capture construction site with workers and heavy machinery; avoid faces.
TARGET_OBJECT: workers, construction site, machinery
RESTRICTIONS: EXCLUDE: faces
Output: hard hat, safety vest, excavator, crane, scaffolding, tools

Example 3
Task: Capture red car in parking lot; reject people.
TARGET_OBJECT: car, vehicle, sedan
RESTRICTIONS: EXCLUDE: people
Output: license plate, tire, windshield, parking lines, headlights
""".strip()


SAM3_MEDIA_LABELS = {
    "photo",
    "photos",
    "video",
    "videos",
    "image",
    "images",
    "picture",
    "pictures",
    "recording",
    "recordings",
    "capture",
    "captures",
}


def _empty_sam3_prompts(status_value: str, error: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status_value,
        "base_prompts": "",
        "extra_prompts": "",
        "prompts": "",
        "target_list": [],
    }
    if error:
        payload["error"] = error
    return payload


def _split_sam3_labels(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    labels: List[str] = []
    for raw in re.split(r"[,;\n]+", text):
        item = re.sub(r"^\s*(?:[-*]|\d+[\).])\s*", "", raw).strip()
        item = re.sub(r"^output:\s*", "", item, flags=re.IGNORECASE).strip()
        item = clean_text(item.strip(" .\"'`"))
        if item and len(item) > 1:
            labels.append(item)
    return labels


def _dedupe_sam3_labels(labels: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for label in labels:
        key = label.lower()
        if key in seen:
            continue
        out.append(label)
        seen.add(key)
    return out


def _clean_sam3_label(label: str) -> str:
    label = clean_text(label)
    label = re.sub(r"^(?:any|all|only|the|a|an)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(
        r"^(?:include|show|capture|submit|upload|provide|use|accept)\s+",
        "",
        label,
        flags=re.IGNORECASE,
    )
    return clean_text(label.strip(" .\"'`"))


def _basic_sam3_label_filter(labels: List[str]) -> List[str]:
    filtered: List[str] = []
    for label in labels:
        item = _clean_sam3_label(label)
        if not item:
            continue
        if item.lower() in SAM3_MEDIA_LABELS:
            continue
        if len(item.split()) > 4:
            continue
        filtered.append(item)
    return filtered


def _extract_sam3_exclusions(description: str, extractions: Dict[str, Optional[str]]) -> List[str]:
    rules_text = " ".join(
        clean_text(value)
        for value in [
            description,
            extractions.get("do_and_donts"),
            extractions.get("safety_privacy"),
        ]
        if value
    )
    if not rules_text:
        return []

    exclusions: List[str] = []
    patterns = [
        r"\b(?:reject|exclude|avoid)\s+(?:any|all)?\s*([^,\.]+?)(?:\.|,|$)",
        r"\bexcluding\s+([^,\.]+?)(?:\.|,|$)",
        r"(?:^|,|\s)no\s+([^,\.]+?)(?:\.|,|$)",
        r"(?:^|,|\s)without\s+([^,\.]+?)(?:\.|,|$)",
        r"\b(?:do not|don't|must not)\s+(?:include|show|capture|submit|upload|accept)?\s*([^,\.]+?)(?:\.|,|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, rules_text, flags=re.IGNORECASE):
            item = _clean_sam3_label(match.group(1) or "")
            if item:
                exclusions.append(item.lower())

    for sentence in re.split(r"[;\.\n]+", rules_text):
        sentence = clean_text(sentence)
        if not sentence:
            continue
        match = re.search(
            r"(.+?)\s+(?:will not be accepted|are not accepted|is not accepted|not accepted|"
            r"will be rejected|are rejected|is rejected|not allowed|are prohibited|is prohibited)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if match:
            item = _clean_sam3_label(match.group(1) or "")
            if item:
                exclusions.append(item.lower())

    return _dedupe_sam3_labels(exclusions)


def _filter_sam3_labels(
    labels: List[str],
    *,
    excluded: List[str],
    base_labels: Optional[List[str]] = None,
) -> List[str]:
    excluded_set = {item.lower() for item in excluded}
    base_set = {item.lower() for item in (base_labels or [])}
    out: List[str] = []
    for label in _basic_sam3_label_filter(labels):
        key = label.lower()
        if key in excluded_set or key in base_set:
            continue
        out.append(label)
    return _dedupe_sam3_labels(out)


def _format_sam3_restrictions(excluded: List[str]) -> str:
    return "EXCLUDE: " + ", ".join(excluded) if excluded else "None"


def _generate_related_sam3_labels(
    *,
    description: str,
    title: str,
    target_object: str,
    restrictions: str,
    backend: OpenRouterBackend,
    max_extra: int = 10,
) -> List[str]:
    user_prompt = f"""{SAM3_RELATED_PROMPTS_FEWSHOT}

Now generate extra labels for this input (follow rules strictly):

Task: {description.strip()}
TITLE: {title.strip() if title.strip() else "None"}
TARGET_OBJECT: {target_object.strip()}
RESTRICTIONS: {restrictions.strip() if restrictions.strip() else "None"}

Output:
"""
    raw = backend.chat(
        system_prompt="You produce segmentation labels for SAM-like text prompts.",
        user_message=user_prompt,
        max_tokens=220,
    )
    return _split_sam3_labels(raw)[:max_extra]


def build_sam3_prompts_for_process(
    *,
    description: str,
    extractions: Dict[str, Optional[str]],
    request_kind: Optional[str],
    backend: OpenRouterBackend,
    title: str = "",
    max_extra: int = 10,
    max_prompts: int = 50,
) -> Dict[str, Any]:
    if _normalize_request_kind(request_kind) == REQUEST_KIND_TEXTUAL:
        return _empty_sam3_prompts("SKIPPED_TEXTUAL_REQUEST")

    target_object = clean_text(extractions.get("target_object") or "")
    if not target_object:
        return _empty_sam3_prompts("SKIPPED_NO_TARGET_OBJECT")

    excluded = _extract_sam3_exclusions(description, extractions)
    restrictions = _format_sam3_restrictions(excluded)
    base_labels = _filter_sam3_labels(
        _split_sam3_labels(target_object),
        excluded=excluded,
    )
    base_prompts = ", ".join(base_labels)

    if not base_labels:
        payload = _empty_sam3_prompts("SKIPPED_NO_USABLE_TARGET_OBJECT")
        return payload

    try:
        generated_extra = _generate_related_sam3_labels(
            description=description,
            title=title,
            target_object=base_prompts,
            restrictions=restrictions,
            backend=backend,
            max_extra=max_extra,
        )
        extra_labels = _filter_sam3_labels(
            generated_extra,
            excluded=excluded,
            base_labels=base_labels,
        )[:max_extra]
        status_value = "COMPLETED"
        error = None
    except Exception as exc:
        extra_labels = []
        status_value = "PARTIAL_RELATED_PROMPTS_FAILED"
        error = str(exc)

    merged_labels = _dedupe_sam3_labels((base_labels + extra_labels)[:max_prompts])
    payload = {
        "status": status_value,
        "base_prompts": base_prompts,
        "extra_prompts": ", ".join(extra_labels),
        "prompts": ", ".join(merged_labels),
        "target_list": merged_labels,
    }
    if error:
        payload["error"] = error
    return payload

# ----- textual_safety.py -----





# Keyword lists shared with the original textual_relevance safety gate.
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

# "shoot"/"shot" are normal camera vocabulary ("use tripod while shooting"),
# so they only count as violence in an explicitly violent context.
UNSAFE_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "violence": [
        r"\bshoot(?:ing|s)?\s+(?:at\s+)?"
        r"(?:someone|somebody|people|person|persons|him|her|them|us|"
        r"civilian|civilians|kid|kids|child|children|crowd|crowds)\b",
        r"\bmass\s+shooting\b",
        r"\bschool\s+shooting\b",
        r"\bshooter\b",
        r"\bgun\s?shot\b",
        r"\bshot\s+(?:dead|to\s+death)\b",
    ],
}


def _is_arabic_text(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in (text or ""))


def _extract_json(text: str) -> Dict[str, Any]:
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


@lru_cache(maxsize=1)
def _unsafe_keyword_matchers() -> Dict[str, List[Tuple[str, Optional[re.Pattern]]]]:
    """Compile the keyword lists once.

    Latin terms match on word boundaries (plus common inflections) so ordinary
    words such as "skills", "establish" or "unisex" are not read as unsafe.
    Arabic terms stay substring-based because of attached prefixes/suffixes.
    """
    matchers: Dict[str, List[Tuple[str, Optional[re.Pattern]]]] = {}
    for category, terms in UNSAFE_CATEGORY_TERMS.items():
        compiled: List[Tuple[str, Optional[re.Pattern]]] = []
        for term in terms:
            if _is_arabic_text(term):
                compiled.append((term, None))
                continue
            escaped = re.escape(term)
            if " " in term:
                suffixes = ""
            else:
                # Allow a doubled final consonant so "stab" also matches "stabbed".
                last = term[-1]
                doubling = f"{re.escape(last)}?" if last.isalpha() and last not in "aeiou" else ""
                suffixes = rf"(?:{doubling}(?:s|es|ed|ing))?"
            compiled.append((term, re.compile(rf"\b{escaped}{suffixes}\b", re.IGNORECASE)))
        for pattern in UNSAFE_CATEGORY_PATTERNS.get(category, []):
            compiled.append(("", re.compile(pattern, re.IGNORECASE)))
        matchers[category] = compiled
    return matchers


def detect_unsafe_keywords(text: str) -> Dict[str, Any]:
    """
    Fast local keyword scan (English + Arabic terms).

    Returns:
        detected (bool), categories (list), terms (list of matched snippets)
    """
    t = (text or "").lower()
    found_categories: List[str] = []
    found_terms: List[str] = []
    for cat, matchers in _unsafe_keyword_matchers().items():
        matched: List[str] = []
        for term, pattern in matchers:
            if pattern is None:
                if term in t:
                    matched.append(term)
                continue
            hit = pattern.search(t)
            if hit:
                matched.append(term or hit.group(0).strip())
        if matched:
            found_categories.append(cat)
            found_terms.extend(matched[:3])
    return {
        "detected": bool(found_categories),
        "categories": found_categories,
        "terms": found_terms[:8],
    }


def _llm_safety_review(
    text: str,
    backend: OpenRouterBackend,
    lang: str,
) -> Dict[str, Any]:
    """Optional OpenRouter pass for contextual unsafe / not-safe-for-work content."""
    system = (
        "You classify whether text is safe for work (SFW) in a professional crowdsourcing platform. "
        "Mark unsafe for: hate speech, racism, sexual content, graphic violence, threats, "
        "harassment, or strong profanity. Neutral opinions and mild language are SFW."
    )
    user = f"""
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
- safe_for_work=YES unless clearly unsafe as defined above.
- Do not flag on-topic professional discussion (e.g. medical, security, news) without harmful intent.
- Write reason in {"Arabic" if lang == "ar" else "English"}.
""".strip()
    try:
        raw = backend.chat(system, user, max_tokens=250)
        obj = _extract_json(raw)
        sfw_raw = str(obj.get("safe_for_work", "YES")).strip().upper()
        is_safe = sfw_raw == "YES"
        try:
            confidence = float(obj.get("confidence", 0.9 if is_safe else 0.85))
        except Exception:
            confidence = 0.9 if is_safe else 0.85
        confidence = max(0.0, min(1.0, confidence))
        categories = obj.get("categories", [])
        if not isinstance(categories, list):
            categories = []
        categories = [str(c).strip() for c in categories if str(c).strip()]
        reason = str(obj.get("reason", "")).strip()
        return {
            "is_safe_for_work": is_safe,
            "confidence": confidence,
            "categories": categories,
            "reason": reason,
            "raw_response": raw,
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
    backend: Optional[OpenRouterBackend] = None,
    use_llm: bool = True,
    output_language: str = "auto",
) -> Dict[str, Any]:
    """
    Check if provider text is safe for work (SFW).

    Args:
        text: Contributor / provider text to inspect.
        use_llm: If True, run OpenRouter review when keyword gate passes.
        output_language: "auto", "en", or "ar" for LLM reason language.

    Returns:
        ok, is_safe_for_work, safe ("YES"/"NO"), score, reason, categories,
        matched_terms, method, unsafe_content (compat shape), model, raw_response.
    """
    effective_model = (
        backend.model_name if backend is not None else _configured_openrouter_model()
    )

    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Empty text"}

    lang = (output_language or "auto").strip().lower()
    if lang == "auto":
        lang = "ar" if _is_arabic_text(text) else "en"

    keyword_hit = detect_unsafe_keywords(text)
    if keyword_hit.get("detected"):
        cats = keyword_hit.get("categories") or []
        cat_str = ", ".join(cats) or "unsafe content"
        return {
            "ok": True,
            "is_safe_for_work": False,
            "safe": "NO",
            "score": 0.0,
            "reason": f"Blocked by keyword safety gate ({cat_str}).",
            "categories": cats,
            "matched_terms": keyword_hit.get("terms") or [],
            "method": "keyword",
            "unsafe_content": keyword_hit,
            "model": effective_model,
            "raw_response": "blocked_by_keyword_gate",
        }

    if not use_llm:
        return {
            "ok": True,
            "is_safe_for_work": True,
            "safe": "YES",
            "score": 1.0,
            "reason": "No unsafe keywords detected (keyword-only mode).",
            "categories": [],
            "matched_terms": [],
            "method": "keyword",
            "unsafe_content": keyword_hit,
            "model": effective_model,
            "raw_response": "",
        }

    if backend is None:
        backend = OpenRouterBackend(temperature=0.0)

    llm = _llm_safety_review(text, backend, lang)
    is_safe = bool(llm.get("is_safe_for_work", True))
    categories = list(llm.get("categories") or [])
    reason = str(llm.get("reason", "")).strip()
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
        "is_safe_for_work": is_safe,
        "safe": "YES" if is_safe else "NO",
        "score": float(llm.get("confidence", 1.0 if is_safe else 0.0)),
        "reason": reason,
        "categories": categories,
        "matched_terms": keyword_hit.get("terms") or [],
        "method": "keyword+llm",
        "unsafe_content": unsafe_content,
        "model": effective_model,
        "raw_response": llm.get("raw_response", ""),
        "llm_error": llm.get("llm_error"),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check if text is safe for work (SFW).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", help="Text to check")
    group.add_argument("--file", "-f", help="Path to a UTF-8 text file")
    parser.add_argument("--no-llm", action="store_true", help="Keyword gate only (no OpenRouter call)")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            content = fh.read()
    else:
        content = args.text or ""

    result = check_text_safety(content, use_llm=not args.no_llm)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif not result.get("ok"):
        print(f"Error: {result.get('error', 'unknown')}", file=sys.stderr)
        return 1
    else:
        print("safe" if result.get("is_safe_for_work") else "unsafe")

    if not result.get("ok"):
        return 1
    return 0 if result.get("is_safe_for_work") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

# ----- target_object_quality.py -----


def _empty_target_object_check(status: str = "SKIPPED") -> Dict[str, Any]:
    return {
        "status": status,
        "is_specific": None,
        "confidence": None,
        "reason": "",
        "category": None,
        "method": None,
        "recommended_question": "",
        "checked_value": "",
        "gate_version": TARGET_OBJECT_QUALITY_VERSION,
    }


def _normalize_token_for_match(token: str) -> str:
    token = clean_text(token).lower()
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _target_object_tokens(text: str) -> List[str]:
    return meaningful_words(text)


def _substantive_target_tokens(target_object: str) -> List[str]:
    tokens = [_normalize_token_for_match(token) for token in _target_object_tokens(target_object)]
    return [token for token in tokens if token not in TARGET_OBJECT_META_TERMS]


def _coherence_tokens(text: str) -> set:
    return {_normalize_token_for_match(token) for token in _target_object_tokens(text)}


def _is_exact_vague_target(target_object: str) -> bool:
    normalized = clean_text(target_object).lower()
    if normalized in VAGUE_TARGET_EXACT:
        return True
    collapsed = re.sub(r"[^\w\s]", " ", normalized)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed in VAGUE_TARGET_EXACT


def _is_meta_only_target(target_object: str) -> bool:
    return len(_substantive_target_tokens(target_object)) == 0


def _target_has_source_coherence(description: str, target_object: str) -> bool:
    substantive = _substantive_target_tokens(target_object)
    if not substantive:
        return False

    source_tokens = _coherence_tokens(description)
    target_tokens = set(substantive)

    if target_tokens & source_tokens:
        return True

    for target_token in target_tokens:
        for source_token in source_tokens:
            if target_token in source_token or source_token in target_token:
                return True

    return False


def _needs_llm_target_review(target_object: str) -> bool:
    substantive = _substantive_target_tokens(target_object)
    if len(substantive) >= 2:
        return False
    if len(substantive) == 1:
        return substantive[0] in SOFT_VAGUE_TARGET_TERMS
    return False


def _target_object_recommended_question(request_kind: Optional[str]) -> str:
    return _field_specs_for_request_kind(request_kind)["target_object"]["question"]


def _heuristic_target_object_check(
    description: str,
    target_object: str,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Dict[str, Any]:
    target = clean_text(target_object)
    question = _target_object_recommended_question(request_kind)

    if not target:
        return {
            **_empty_target_object_check("MISSING"),
            "is_specific": False,
            "reason": "No target object was extracted.",
            "category": "missing",
            "method": "heuristic",
            "recommended_question": question,
            "checked_value": "",
        }

    if _is_exact_vague_target(target):
        return {
            **_empty_target_object_check("VAGUE"),
            "is_specific": False,
            "confidence": 1.0,
            "reason": (
                f"The extracted target '{target}' is a placeholder/meta term, not a concrete subject."
            ),
            "category": "placeholder_meta_term",
            "method": "heuristic",
            "recommended_question": question,
            "checked_value": target,
        }

    if _is_meta_only_target(target):
        return {
            **_empty_target_object_check("VAGUE"),
            "is_specific": False,
            "confidence": 1.0,
            "reason": (
                f"The extracted target '{target}' contains only generic/meta words, not a real subject."
            ),
            "category": "meta_terms_only",
            "method": "heuristic",
            "recommended_question": question,
            "checked_value": target,
        }

    if not _target_has_source_coherence(description, target):
        return {
            **_empty_target_object_check("VAGUE"),
            "is_specific": False,
            "confidence": 0.95,
            "reason": (
                f"The extracted target '{target}' is not grounded in the task description."
            ),
            "category": "not_grounded_in_source",
            "method": "heuristic",
            "recommended_question": question,
            "checked_value": target,
        }

    if _needs_llm_target_review(target):
        return {
            **_empty_target_object_check("REVIEW"),
            "is_specific": None,
            "confidence": None,
            "reason": "Target object needs semantic review.",
            "category": "needs_llm_review",
            "method": "heuristic",
            "recommended_question": question,
            "checked_value": target,
        }

    return {
        **_empty_target_object_check("SPECIFIC"),
        "is_specific": True,
        "confidence": 0.9,
        "reason": "Target object passed heuristic specificity checks.",
        "category": "specific",
        "method": "heuristic",
        "recommended_question": question,
        "checked_value": target,
    }


def _target_object_specificity_prompt(
    description: str,
    target_object: str,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Tuple[str, str]:
    request_kind = _normalize_request_kind(request_kind)
    subject_label = "visual subject" if request_kind == REQUEST_KIND_VISUAL else "text topic/content"

    system_prompt = f"""You judge whether an extracted target_object is specific enough for a crowdsourcing task.

Return ONLY valid JSON with this schema:
{{
  "is_specific": true,
  "confidence": 0.0,
  "reason": "short reason",
  "category": "specific|placeholder_meta_term|too_generic|template_boilerplate|not_actionable"
}}

Definitions:
- specific: a concrete {subject_label} a contributor can understand and act on
  (e.g. cats, storefronts, construction workers wearing helmets, recycling in schools).
- not specific: meta placeholders, template boilerplate, or generic words without a real subject
  (e.g. main subject, object, thing, event, content, the answer, the document).

Rules:
- Reject meta references like "main subject", "subject", "object", "thing", "something".
- Reject template/system wording that does not name what to capture or write about.
- Accept short but concrete subjects, including single common nouns like people, cats, cars, art.
- Do not infer missing details; judge only what is present in the extracted target_object.
"""
    user_prompt = (
        f'Task description:\n"{description}"\n\n'
        f"Extracted target_object:\n\"{target_object}\"\n\n"
        f"Request type: {request_kind}\n\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def detect_target_object_specificity(
    description: str,
    target_object: str,
    backend: OpenRouterBackend,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Dict[str, Any]:
    request_kind = _normalize_request_kind(request_kind)
    target = clean_text(target_object)
    question = _target_object_recommended_question(request_kind)
    system_prompt, user_prompt = _target_object_specificity_prompt(
        description,
        target,
        request_kind,
    )

    try:
        raw = backend.chat(system_prompt=system_prompt, user_message=user_prompt, max_tokens=180)
        parsed = try_parse_json(raw) or {}
        is_specific = parsed.get("is_specific")
        if isinstance(is_specific, str):
            is_specific = is_specific.strip().lower() in {"1", "true", "yes"}
        else:
            is_specific = bool(is_specific)

        confidence_raw = parsed.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.85 if is_specific else 0.85

        reason = clean_text(str(parsed.get("reason") or ""))
        category = clean_text(str(parsed.get("category") or ("specific" if is_specific else "too_generic")))

        if not reason:
            reason = (
                "Target object is specific enough for contributors."
                if is_specific
                else "Target object is too vague or generic for contributors."
            )

        return {
            "status": "SPECIFIC" if is_specific else "VAGUE",
            "is_specific": is_specific,
            "confidence": confidence,
            "reason": reason,
            "category": category or ("specific" if is_specific else "too_generic"),
            "method": "llm",
            "recommended_question": "" if is_specific else question,
            "checked_value": target,
            "gate_version": TARGET_OBJECT_QUALITY_VERSION,
            "raw_response": raw,
        }
    except Exception as exc:
        return {
            **_empty_target_object_check("ERROR"),
            "is_specific": False,
            "confidence": 0.0,
            "reason": f"Target object review failed: {exc}",
            "category": "review_error",
            "method": "llm",
            "recommended_question": question,
            "checked_value": target,
        }


def validate_target_object_quality(
    description: str,
    target_object: Optional[str],
    backend: OpenRouterBackend,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Dict[str, Any]:
    request_kind = _normalize_request_kind(request_kind)
    target = clean_text(target_object or "")

    if not target:
        return _heuristic_target_object_check(description, "", request_kind)

    heuristic = _heuristic_target_object_check(description, target, request_kind)
    if heuristic.get("status") in {"VAGUE", "MISSING"}:
        return heuristic

    if heuristic.get("status") == "REVIEW":
        llm_result = detect_target_object_specificity(
            description,
            target,
            backend,
            request_kind=request_kind,
        )
        if llm_result.get("is_specific") is False:
            return llm_result
        if llm_result.get("is_specific") is True:
            return llm_result

    return heuristic


# ----- readiness.py -----


def detect_task_safety_risks(description: str, backend: OpenRouterBackend) -> Dict[str, Any]:
    description = clean_text(description)
    if not description:
        return {
            "safe": False,
            "detected": True,
            "categories": [],
            "terms": [],
            "reason": "Empty text",
            "user_message": "Please provide a clear work-task description before validating.",
            "checked_by": "textual_safety",
            "method": None,
            "score": 0.0,
        }

    result = check_text_safety(
        description,
        backend=backend,
        use_llm=True,
    )

    if not result.get("ok"):
        err = str(result.get("error", "Safety check failed."))
        return {
            "safe": False,
            "detected": True,
            "categories": [],
            "terms": [],
            "reason": err,
            "user_message": err,
            "checked_by": "textual_safety",
            "method": None,
            "score": 0.0,
        }

    is_safe = bool(result.get("is_safe_for_work", False))
    user_message = (
        UNSAFE_FOR_WORK_RECOMMENDED_MESSAGE
        if not is_safe
        else str(result.get("reason", "")).strip()
    )
    return {
        "safe": is_safe,
        "detected": not is_safe,
        "categories": list(result.get("categories") or []),
        "terms": list(result.get("matched_terms") or []),
        "reason": str(result.get("reason", "")).strip(),
        "user_message": user_message,
        "checked_by": "textual_safety",
        "method": result.get("method"),
        "score": result.get("score"),
        "raw_response": result.get("raw_response", ""),
    }


def _missing_required_labels(
    extractions: Dict[str, Optional[str]],
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> List[str]:
    return [
        label
        for label in _required_labels_for_request_kind(request_kind)
        if not extractions.get(label)
    ]


def _safety_is_ok(safety_check: Dict[str, Any]) -> bool:
    """True only when safety_check explicitly reports safe."""
    return safety_check.get("safe") is True


def _safety_check_payload(safety: Dict[str, Any]) -> Dict[str, Any]:
    is_safe = safety.get("safe") is True
    user_message = (
        UNSAFE_FOR_WORK_RECOMMENDED_MESSAGE
        if not is_safe
        else str(safety.get("user_message") or safety.get("reason") or "").strip()
    )
    return {
        "status": "SAFE" if is_safe else "UNSAFE",
        "safe": is_safe,
        "categories": list(safety.get("categories") or []),
        "terms": list(safety.get("terms") or []),
        "reason": safety.get("reason", ""),
        "user_message": user_message,
        "method": safety.get("method"),
        "score": safety.get("score"),
        "checked_by": safety.get("checked_by", "textual_safety"),
    }


def evaluate_ready_for_work(
    description: str,
    extractions: Dict[str, Optional[str]],
    safety: Dict[str, Any],
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
    target_object_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request_kind = _normalize_request_kind(request_kind)
    questions = _questions_for_request_kind(request_kind)
    missing_required = _missing_required_labels(extractions, request_kind)
    is_safe = safety.get("safe") is True

    blockers: List[str] = []
    if not is_safe:
        cats = ", ".join(safety.get("categories", []) or [])
        blockers.append(f"Safety gate failed: {cats or 'unsafe content'}.")
    if is_safe and missing_required:
        blockers.append("Missing required work fields: " + ", ".join(missing_required) + ".")

    target_check = target_object_check or {}
    target_is_specific = target_check.get("is_specific")
    if (
        is_safe
        and extractions.get("target_object")
        and target_is_specific is False
        and "target_object" not in missing_required
    ):
        missing_required = missing_required + ["target_object"]
        reason = clean_text(str(target_check.get("reason") or ""))
        blockers.append(
            "Target object is too vague: " + (reason or "please name a concrete subject.")
        )

    if not is_safe:
        suggestions = [UNSAFE_FOR_WORK_RECOMMENDED_MESSAGE]
    else:
        suggestions = []
        for label in missing_required:
            suggestions.append(questions[label])
        if target_is_specific is False:
            vague_question = clean_text(str(target_check.get("recommended_question") or ""))
            if vague_question:
                suggestions.append(vague_question)
        for label in _optional_labels_for_request_kind(request_kind):
            if not extractions.get(label):
                suggestions.append(questions[label])

    ready = is_safe and len(missing_required) == 0
    return {
        "ready_for_work": ready,
        "ready_for_enhancement": ready,
        "status": "READY" if ready else "NOT_READY",
        "request_type": request_kind,
        "missing_required_labels": missing_required,
        "blockers": blockers,
        "suggestions": suggestions if not is_safe else dedupe(suggestions),
    }


def validate_before_work(
    description: str,
    backend: OpenRouterBackend,
) -> Tuple[Dict[str, Any], str, Dict[str, Optional[str]]]:
    """Stage 1 safety -> request kind check -> extraction (if safe) -> readiness."""
    description = clean_text(description)
    safety = detect_task_safety_risks(description, backend)
    is_safe = safety.get("safe") is True

    raw_extraction = ""
    if is_safe:
        request_type_check = detect_request_kind(description, backend)
        request_kind = _normalize_request_kind(request_type_check.get("request_type"))
        raw_extraction, extractions = extract_fields(description, backend, request_kind)
        extractions = sanitize_extractions_for_source(description, extractions, request_kind)
        extraction_status = "COMPLETED"
    else:
        request_type_check = _skipped_request_kind_check()
        request_kind = REQUEST_KIND_VISUAL
        extractions = {label: None for label in LABELS}
        extraction_status = "SKIPPED_UNSAFE"

    missing_required = _missing_required_labels(extractions, request_kind)
    if not is_safe:
        required_status = "SKIPPED_UNSAFE"
        target_object_check = _empty_target_object_check("SKIPPED_UNSAFE")
    else:
        target_object_check = validate_target_object_quality(
            description,
            extractions.get("target_object"),
            backend,
            request_kind=request_kind,
        )
        if (
            extractions.get("target_object")
            and target_object_check.get("is_specific") is False
            and "target_object" not in missing_required
        ):
            missing_required = missing_required + ["target_object"]

        if not missing_required:
            required_status = "COMPLETE"
        else:
            required_status = "INCOMPLETE"

    readiness = evaluate_ready_for_work(
        description,
        extractions,
        safety,
        request_kind,
        target_object_check=target_object_check,
    )
    can_enhance = bool(readiness.get("ready_for_enhancement"))

    validation = {
        "safety": safety,
        "safety_check": _safety_check_payload(safety),
        "request_type_check": request_type_check,
        "target_object_check": target_object_check,
        "extraction": {
            "status": extraction_status,
            "ran": is_safe,
            "request_type": request_kind if is_safe else None,
            "prompt_profile": request_type_check.get("prompt_profile"),
            "extractions": extractions,
            "raw_extraction": raw_extraction,
        },
        "required_fields": {
            "status": required_status,
            "labels": _required_labels_for_request_kind(request_kind),
            "missing_required_labels": missing_required,
            "ready_for_enhancement": can_enhance and is_safe,
            "target_object_ok": target_object_check.get("is_specific") is not False,
        },
        "readiness": readiness,
        "can_continue_pipeline": can_enhance,
        "can_enhance": can_enhance,
        "safety_checked_on": "original_description",
        "gate_version": SAFETY_GATE_VERSION,
        "request_kind_check_version": REQUEST_KIND_CHECK_VERSION,
    }
    return validation, raw_extraction, extractions

# ----- enhancement.py -----





def build_draft_from_extractions(extractions: Dict[str, Optional[str]]) -> str:
    """
    Deterministic reformulation from extracted labels (same pattern as approved UI pipeline).
    Prefer a natural visual-media sentence: action + media phrase + subject + location.
    """
    action = extractions.get("action_required")
    target = extractions.get("target_object")
    media_type = extractions.get("media_type")
    media_quantity = extractions.get("media_quantity")
    location = extractions.get("location_details")
    time_details = extractions.get("time_details")
    reward = extractions.get("reward")
    quantity_per_contributor = extractions.get("quantity_per_contributor")
    worker_requirements = extractions.get("worker_requirements")
    camera_specs = extractions.get("camera_specs")
    do_and_donts = extractions.get("do_and_donts")
    safety_privacy = extractions.get("safety_privacy")

    sentences = []
    main_parts = []

    if action:
        main_parts.append(action.strip().capitalize())
    if media_type:
        if target:
            main_parts.append(f"{media_type.strip()} showing {target.strip()}")
        else:
            main_parts.append(media_type.strip())
    elif target:
        main_parts.append(target.strip())
    if media_quantity:
        main_parts.append(f"with a total quantity of {media_quantity.strip()}")
    if location:
        main_parts.append(f"in {location.strip()}")
    if main_parts:
        sentences.append(" ".join(main_parts) + ".")
    if time_details:
        sentences.append(f"The task should be completed during {time_details.strip()}.")
    if quantity_per_contributor:
        sentences.append(f"Each contributor should provide {quantity_per_contributor.strip()}.")
    if reward:
        sentences.append(f"The reward is {reward.strip()}.")
    if worker_requirements:
        sentences.append(f"Workers must meet these requirements: {worker_requirements.strip()}.")
    if camera_specs:
        sentences.append(f"Camera or device requirements: {camera_specs.strip()}.")
    if do_and_donts:
        sentences.append(f"Follow these rules and restrictions: {do_and_donts.strip()}.")
    if safety_privacy:
        sentences.append(f"Respect these safety and privacy constraints: {safety_privacy.strip()}.")

    return clean_text(" ".join(sentences))


def build_textual_draft_from_extractions(extractions: Dict[str, Optional[str]]) -> str:
    """
    Deterministic reformulation for textual requests.
    target_object is the content/topic; action_required is what was asked for.
    """
    action = extractions.get("action_required")
    target = extractions.get("target_object")
    media_type = extractions.get("media_type")
    media_quantity = extractions.get("media_quantity")
    location = extractions.get("location_details")
    time_details = extractions.get("time_details")
    reward = extractions.get("reward")
    quantity_per_contributor = extractions.get("quantity_per_contributor")
    worker_requirements = extractions.get("worker_requirements")
    camera_specs = extractions.get("camera_specs")
    do_and_donts = extractions.get("do_and_donts")
    safety_privacy = extractions.get("safety_privacy")

    sentences = []
    main_parts = []

    if action:
        main_parts.append(action.strip().capitalize())
    if target:
        main_parts.append(target.strip())
    if main_parts:
        sentences.append(" ".join(main_parts) + ".")
    if media_type:
        sentences.append(f"The requested textual format is {media_type.strip()}.")
    if media_quantity:
        sentences.append(f"The requested amount is {media_quantity.strip()}.")
    if location:
        sentences.append(f"Use the relevant location context: {location.strip()}.")
    if time_details:
        sentences.append(f"The task should be completed during {time_details.strip()}.")
    if quantity_per_contributor:
        sentences.append(f"Each contributor may provide {quantity_per_contributor.strip()}.")
    if reward:
        sentences.append(f"The reward is {reward.strip()}.")
    if worker_requirements:
        sentences.append(f"Workers must meet these requirements: {worker_requirements.strip()}.")
    if camera_specs:
        sentences.append(f"Text or document requirements: {camera_specs.strip()}.")
    if do_and_donts:
        sentences.append(f"Follow these rules and restrictions: {do_and_donts.strip()}.")
    if safety_privacy:
        sentences.append(f"Respect these safety and privacy constraints: {safety_privacy.strip()}.")

    return clean_text(" ".join(sentences))


SHORT_TEXT_MODIFIERS = (
    "short",
    "brief",
    "concise",
    "compact",
)
LONG_TEXT_MODIFIERS = (
    "long",
    "lengthy",
    "detailed",
    "in-depth",
    "in depth",
    "comprehensive",
    "extensive",
    "elaborate",
)
TEXTUAL_FORMAT_NOUNS = (
    "review",
    "summary",
    "report",
    "essay",
    "article",
    "document",
    "answer",
    "response",
    "paragraph",
    "point of view",
    "opinion",
    "email",
    "caption",
)


def _has_word_or_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE) is not None


def _contains_any_phrase(text: str, phrases: Tuple[str, ...]) -> bool:
    return any(_has_word_or_phrase(text, phrase) for phrase in phrases)


def _text_length_modifier_hits(text: str) -> List[Tuple[int, str, str]]:
    hits: List[Tuple[int, str, str]] = []
    for group, phrases in (
        ("short", SHORT_TEXT_MODIFIERS),
        ("long", LONG_TEXT_MODIFIERS),
    ):
        for phrase in phrases:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", text or "", flags=re.IGNORECASE):
                hits.append((match.start(), group, phrase))
    return sorted(hits, key=lambda item: item[0])


def _preferred_text_length_modifier(text: str) -> Optional[Tuple[str, str]]:
    hits = _text_length_modifier_hits(text)
    if not hits:
        return None
    _, group, phrase = hits[0]
    return group, phrase


def _source_has_conflicting_text_length_modifiers(text: str) -> bool:
    groups = {group for _, group, _ in _text_length_modifier_hits(text)}
    return "short" in groups and "long" in groups


def _cleanup_text_after_modifier_removal(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([/])\s+", r" \1 ", text)
    text = re.sub(r"\b(?:a|an)\s+(one|1|single)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s+([aeiou])", r"an \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban\s+([^aeiou\s])", r"a \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def _remove_unsupported_opposing_modifiers(value: str, source: str) -> str:
    out = clean_text(value)
    if not out:
        return out

    preferred = _preferred_text_length_modifier(source)
    if not preferred:
        return out

    preferred_group, preferred_phrase = preferred
    source_has_conflict = _source_has_conflicting_text_length_modifiers(source)
    phrases_to_remove: List[str] = []
    for group, phrases in (
        ("short", SHORT_TEXT_MODIFIERS),
        ("long", LONG_TEXT_MODIFIERS),
    ):
        for phrase in phrases:
            if phrase == preferred_phrase:
                continue
            is_opposing_group = group != preferred_group
            is_generated_same_group = group == preferred_group and not _has_word_or_phrase(source, phrase)
            is_later_modifier_in_conflict = source_has_conflict
            if is_opposing_group or is_generated_same_group or is_later_modifier_in_conflict:
                phrases_to_remove.append(phrase)

    for phrase in phrases_to_remove:
        out = re.sub(
            rf"(?:\s*(?:,|/|\b(?:and|or|but)\b)\s*)?\b{re.escape(phrase)}\b\s*",
            " ",
            out,
            flags=re.IGNORECASE,
        )

    return _cleanup_text_after_modifier_removal(out)


def _source_textual_action(description: str) -> Optional[str]:
    action_patterns = [
        r"\bgive\s+a\s+point\s+of\s+view\b",
        r"\b(write|answer|explain|summarize|summarise|review|translate|draft|create|generate|describe|list|compare|argue|analyze|analyse)\b",
    ]
    for pattern in action_patterns:
        match = re.search(pattern, description or "", flags=re.IGNORECASE)
        if match:
            action = match.group(0).strip()
            return action[:1].upper() + action[1:]
    return None


def _source_textual_format_phrase(description: str) -> Optional[str]:
    nouns = "|".join(re.escape(noun) for noun in TEXTUAL_FORMAT_NOUNS)
    modifiers = (
        "short|brief|concise|compact|long|lengthy|detailed|in-depth|in depth|"
        "comprehensive|extensive|elaborate|one-page|single-page|\\d+-page"
    )
    pattern = rf"\b(?:(?:{modifiers})\s+)?(?:{nouns})\b"
    for match in re.finditer(pattern, description or "", flags=re.IGNORECASE):
        phrase = _remove_unsupported_opposing_modifiers(match.group(0).strip(), description)
        preferred = _preferred_text_length_modifier(description)
        if (
            preferred
            and _source_has_conflicting_text_length_modifiers(description)
            and not _contains_any_phrase(phrase, SHORT_TEXT_MODIFIERS + LONG_TEXT_MODIFIERS)
        ):
            for noun in TEXTUAL_FORMAT_NOUNS:
                if _has_word_or_phrase(phrase, noun):
                    phrase = f"{preferred[1]} {noun}"
                    break
        if phrase.lower() in {"paragraph", "answer", "response"}:
            if not _source_has_conflicting_text_length_modifiers(description):
                continue
        return phrase
    return None


def _source_textual_quantity_phrase(description: str) -> Optional[str]:
    patterns = [
        r"\b(?:one|1|single)\s+paragraph\b",
        r"\b\d+\s*(?:-|to)\s*\d+\s+(?:words|paragraphs|pages|bullet points|points|answers|documents)\b",
        r"\b\d+\s+(?:words|paragraphs|pages|bullet points|points|answers|documents)\b",
        r"\b(?:one|1|single)-page\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, description or "", flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _replace_unsupported_generated_format_nouns(text: str, source: str) -> str:
    source_lower = (source or "").lower()
    out = text

    source_primary = None
    for noun in TEXTUAL_FORMAT_NOUNS:
        if _has_word_or_phrase(source_lower, noun):
            source_primary = noun
            break

    if source_primary and source_primary != "account" and not _has_word_or_phrase(source_lower, "account"):
        out = re.sub(r"\baccount\b", source_primary, out, flags=re.IGNORECASE)

    quantity = _source_textual_quantity_phrase(source)
    if quantity and "paragraph" in quantity.lower():
        out = re.sub(
            r"\b(?:a|an|one|single)?\s*(?:short|brief|concise|compact|long|lengthy|detailed|in-depth|comprehensive|extensive|elaborate)?\s*paragraph\b",
            quantity,
            out,
            flags=re.IGNORECASE,
        )

    return _cleanup_text_after_modifier_removal(out)


_PURPOSE_STOPWORDS = {
    "the",
    "our",
    "their",
    "your",
    "a",
    "an",
    "this",
    "that",
    "requested",
    "purpose",
    "purposes",
    "data",
    "collection",
    "research",
    "study",
    "training",
    "needed",
    "for",
    "request",
}


def _strip_ungrounded_purpose_suffix(source: str, text: str) -> str:
    """Drop trailing 'for ...' clauses the model invented (not present in the source)."""
    out = clean_text(text)
    if not out:
        return out

    match = re.search(
        r"[\s,]+((?:needed\s+)?for\s+.{3,80})$",
        out,
        flags=re.IGNORECASE,
    )
    if not match:
        return out

    clause = match.group(1).strip().rstrip(".")
    if re.search(
        r"\b(photo|photos|image|images|video|videos|clip|clips|word|words)\b",
        clause,
        flags=re.IGNORECASE,
    ):
        return out

    source_lower = f" {clean_text(source).lower()} "
    clause_core = re.sub(r"^(?:needed\s+)?for\s+", "", clause, flags=re.IGNORECASE)
    clause_core = re.sub(r"\s+", " ", clause_core).strip().lower()
    if clause_core and clause_core in source_lower:
        return out

    clause_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", clause)
        if token.lower() not in _PURPOSE_STOPWORDS
    ]
    if clause_tokens and any(f" {token} " in source_lower for token in clause_tokens):
        return out

    stripped = clean_text(out[: match.start()]).rstrip(" ,;")
    if stripped and stripped[-1] not in ".!?":
        stripped += "."
    return stripped


_EXAMPLE_CLAUSE_STOPWORDS = {
    "and",
    "the",
    "or",
    "such",
    "as",
    "including",
    "include",
    "includes",
    "other",
    "others",
    "etc",
    "equipment",
    "assets",
    "items",
    "types",
}


def _strip_ungrounded_example_clause(source: str, text: str) -> str:
    """Drop 'including X, Y, or Z' lists the model invented (not present in the source)."""
    out = clean_text(text)
    if not out:
        return out

    source_lower = f" {clean_text(source).lower()} "
    pattern = re.compile(
        r"[\s,]*\b(?:including|such\s+as|for\s+example|e\.g\.)\b([^.;]*)",
        flags=re.IGNORECASE,
    )
    while True:
        match = pattern.search(out)
        if not match:
            return out

        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", match.group(1))
            if token.lower() not in _EXAMPLE_CLAUSE_STOPWORDS
        ]
        if not tokens or any(f" {token} " in source_lower for token in tokens):
            return out

        out = clean_text(out[: match.start()] + " " + out[match.end():])
        out = re.sub(r"\s+([,.;])", r"\1", out).strip(" ,;")
        if out and out[-1] not in ".!?":
            out += "."


_WORD_NUMBERS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "hundred": "100",
}
_STILL_MEDIA_WORDS = (
    "photo",
    "photos",
    "photograph",
    "photographs",
    "picture",
    "pictures",
    "image",
    "images",
    "snapshot",
    "snapshots",
    "صورة",
    "صور",
)
_MOTION_MEDIA_WORDS = (
    "video",
    "videos",
    "clip",
    "clips",
    "footage",
    "recording",
    "recordings",
    "فيديو",
    "مقطع",
    "مقاطع",
)


def _numbers_in(text: str) -> set:
    lowered = (text or "").lower()
    found = set(re.findall(r"\d+", lowered))
    for word, digits in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", lowered):
            found.add(digits)
    return found


def _mentions_any_word(text: str, words: Tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", lowered) for word in words)


def polish_faithfulness_issues(source: str, polished: str) -> List[str]:
    """Report content the polished text added but the source never mentioned."""
    if not clean_text(polished):
        return ["empty_polished_output"]

    issues: List[str] = []

    invented_numbers = _numbers_in(polished) - _numbers_in(source)
    if invented_numbers:
        issues.append("invented_numbers:" + ",".join(sorted(invented_numbers)))

    for label, words in (("still_media", _STILL_MEDIA_WORDS), ("motion_media", _MOTION_MEDIA_WORDS)):
        if _mentions_any_word(polished, words) and not _mentions_any_word(source, words):
            issues.append(f"invented_{label}")

    return issues


def enforce_source_coherence(
    source: str,
    text: str,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> str:
    out = _remove_unsupported_opposing_modifiers(text, source)
    if _normalize_request_kind(request_kind) == REQUEST_KIND_TEXTUAL:
        out = _replace_unsupported_generated_format_nouns(out, source)
    out = _strip_ungrounded_example_clause(source, out)
    out = _strip_ungrounded_purpose_suffix(source, out)
    return clean_text(out)


def sanitize_extractions_for_source(
    description: str,
    extractions: Dict[str, Optional[str]],
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Dict[str, Optional[str]]:
    sanitized = dict(extractions)
    for count_label in ("media_quantity", "quantity_per_contributor"):
        value = sanitized.get(count_label)
        if value:
            sanitized[count_label] = _normalize_media_count_value(value)

    if _normalize_request_kind(request_kind) != REQUEST_KIND_TEXTUAL:
        return sanitized

    for key, value in list(sanitized.items()):
        if isinstance(value, str):
            sanitized[key] = _remove_unsupported_opposing_modifiers(value, description)

    action = _source_textual_action(description)
    if action:
        sanitized["action_required"] = action

    source_format = _source_textual_format_phrase(description)
    if source_format:
        sanitized["media_type"] = source_format

    source_quantity = _source_textual_quantity_phrase(description)
    if source_quantity:
        sanitized["media_quantity"] = _normalize_media_count_value(source_quantity)

    return sanitized


def build_draft_for_request_kind(
    extractions: Dict[str, Optional[str]],
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> str:
    if _normalize_request_kind(request_kind) == REQUEST_KIND_TEXTUAL:
        return build_textual_draft_from_extractions(extractions)
    return build_draft_from_extractions(extractions)


_SHORT_GREETING_RE = re.compile(
    r"^\s*((?:hello|hi|hey(?:\s+there)?|dear|greetings|"
    r"good\s+morning|good\s+afternoon|good\s+evening)"
    r"(?:\s+\w+){0,3})\s*[,:!]*",
    flags=re.IGNORECASE,
)
_TASK_LIKE_FRAGMENT_RE = re.compile(
    r"\b("
    r"write|send|provide|capture|take|submit|upload|record|describe|explain|"
    r"please|need your|we need|images?|photos?|videos?|words?"
    r")\b",
    flags=re.IGNORECASE,
)


def _fragment_is_bulk_of_source(fragment: str, original: str) -> bool:
    fragment_norm = clean_text(fragment).lower()
    original_norm = clean_text(original).lower()
    if not fragment_norm or not original_norm:
        return False
    if fragment_norm == original_norm:
        return True
    return len(fragment_norm) >= max(40, int(0.45 * len(original_norm)))


def _requester_facing_source_fragments(description: str) -> List[str]:
    original = clean_text(description)
    if not original:
        return []

    fragments: List[str] = []
    greeting_match = _SHORT_GREETING_RE.match(original)
    if greeting_match:
        fragments.append(greeting_match.group(1).strip())

    org_terms = (
        "company|brand|team|startup|organization|organisation|agency|business|client|"
        "project|campaign|product|app|platform|association|charity|nonprofit|non-profit|"
        "ngo|foundation|shelter|rescue|clinic|care group|research group|institute|lab"
    )
    context_patterns = [
        rf"\bwe\s+are\s+(a|an|the|our)\s+({org_terms})\b",
        rf"\b(my|our|the)\s+({org_terms})\b",
        rf"\b({org_terms})\s+(called|named|is)\b",
        rf"\bfor\s+(my|our|the)\s+({org_terms})\b",
        r"\bour\s+(mission|goal|initiative|research|study|campaign|rescue|care work|welfare work)\b",
        rf"\bwe\s+(run|operate|own|represent|support)\s+(a|an|the|our)\s+({org_terms})\b",
        r"\bwe\s+are\s+a\s+company\b",
        r"\bwe\s+are\s+an\s+organization\b",
        r"\bwe\s+are\s+an\s+organisation\b",
        r"\bwe\s+are\s+operating\b",
        r"\bwe\s+aim\s+to\b",
    ]
    purpose_patterns = [
        r"\b(reason|purpose|goal|mission|initiative|research|study|campaign)\b",
        r"\b(to help|to support|to document|to identify|to assess|to evaluate|to collect|to gather|to understand|to improve|so we can|so that we can|because we need to|in order to)\b",
        r"\bneed(s|ed)?\s+(assistance|help|care|rescue|treatment|support)\b",
        r"\b(injur(y|ies|ed)|wound(s|ed)?|poor health|sick|ill|stray|abandoned|neglected)\b",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", original)
    seen = {fragment.lower() for fragment in fragments}
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        if _fragment_is_bulk_of_source(sentence, original):
            continue
        if _TASK_LIKE_FRAGMENT_RE.search(sentence):
            continue
        if not (
            any(re.search(pattern, lower) for pattern in context_patterns)
            or any(re.search(pattern, lower) for pattern in purpose_patterns)
        ):
            continue
        key = lower
        if key in seen:
            continue
        fragments.append(sentence)
        seen.add(key)

    return fragments[:5]


def build_requester_facing_from_source(description: str, polished: str) -> Tuple[str, bool]:
    polished = clean_text(polished)
    original = clean_text(description)
    fragments = _requester_facing_source_fragments(description)
    fragments = [
        cleaned
        for cleaned in (_remove_unsupported_opposing_modifiers(fragment, description) for fragment in fragments)
        if cleaned
        and not _fragment_is_bulk_of_source(cleaned, original)
        and not _SHORT_GREETING_RE.fullmatch(cleaned.rstrip(" ,:!;."))
    ]
    if not fragments:
        return polished, True

    combined = clean_text(" ".join(fragments + [polished]))
    original_norm = original.lower()
    combined_norm = combined.lower()
    polished_norm = polished.lower()
    if (
        original_norm
        and combined_norm.startswith(original_norm)
        and polished_norm not in original_norm
    ):
        return polished, True
    return combined, False


def run_enhancement_pipeline(
    description: str,
    backend: OpenRouterBackend,
    *,
    extractions: Optional[Dict[str, Optional[str]]] = None,
    raw_original: Optional[str] = None,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
) -> Tuple[str, Dict[str, Any]]:
    """Draft from extractions -> LLM polish. Re-extracts unless extractions are provided."""
    description = clean_text(description)
    if not description:
        return "", {"ok": False, "error": "Empty description.", "final_description": ""}

    request_kind = _normalize_request_kind(request_kind)
    questions = _questions_for_request_kind(request_kind)

    if extractions is None:
        raw_original, extractions = extract_fields(description, backend, request_kind)
    else:
        raw_original = raw_original or ""
    extractions = sanitize_extractions_for_source(description, extractions, request_kind)

    missing_required = _missing_required_labels(extractions, request_kind)
    if missing_required:
        return "", {
            "ok": False,
            "error": "Missing required fields for enhancement: " + ", ".join(missing_required) + ".",
            "missing_required_labels": missing_required,
            "missing_questions": {label: questions[label] for label in missing_required},
            "final_description": "",
            "request_type": request_kind,
        }

    draft = build_draft_for_request_kind(extractions, request_kind)

    if not draft:
        missing_labels = top_missing_labels(extractions, k=5, request_kind=request_kind)
        return "", {
            "ok": False,
            "error": "No usable extracted fields found.",
            "raw_input": description,
            "raw_original_extraction": raw_original,
            "original_extractions": extractions,
            "missing_labels": missing_labels,
            "missing_questions": {label: questions[label] for label in missing_labels},
            "final_description": "",
            "request_type": request_kind,
        }

    polished = enforce_source_coherence(
        description,
        clean_text(backend.polish_description(draft, description)),
        request_kind,
    )
    final_source = "polished"
    issues = polish_faithfulness_issues(description, polished)

    if issues:
        retried = enforce_source_coherence(
            description,
            clean_text(backend.polish_description(draft, description, strict=True)),
            request_kind,
        )
        retry_issues = polish_faithfulness_issues(description, retried)
        if not retry_issues:
            polished, issues, final_source = retried, [], "polished_strict_retry"
        else:
            issues = retry_issues

    if issues:
        polished = enforce_source_coherence(description, draft, request_kind)
        final_source = "deterministic_draft"

    final_description = polished or clean_text(draft)
    requester_facing, requester_facing_skipped = build_requester_facing_from_source(
        description,
        final_description,
    )

    return final_description, {
        "ok": True,
        "raw_input": description,
        "request_type": request_kind,
        "raw_original_extraction": raw_original,
        "original_extractions": extractions,
        "draft_from_extractions": draft,
        "polished_description": final_description,
        "requester_facing_description": requester_facing,
        "requester_facing_skipped": requester_facing_skipped,
        "requester_facing_error": None,
        "final_source": final_source,
        "polish_faithfulness_issues": issues,
        "final_description": final_description,
        "missing_labels": top_missing_labels(extractions, k=5, request_kind=request_kind),
    }

# ----- local_titles.py -----




TitleMode = Literal["zero_shot", "few_shot_cot"]

_DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch is not None
    else "unavailable"
)

_CHECKPOINT_DIR = Path(TITLE_GEN_DIR)

_tokenizer = None
_model = None

TITLE_FEW_SHOT_EXAMPLES = [
    ("Capture images of ducks swimming at night in a pond.", "Ducks at Night"),
    ("Record video of retail shelves and validate coverage quality.", "Retail Shelves"),
    (
        "Take photos of construction workers wearing safety helmets on a building site.",
        "Workers Wearing Helmets",
    ),
    ("Film cyclists riding through an urban intersection during rush hour.", "Cyclists at Intersection"),
]

_TITLE_FEW_SHOT_BLOCK = "\n".join(
    f"Description: {d}\nTitle: {t}" for d, t in TITLE_FEW_SHOT_EXAMPLES
)

INSTRUCTION = (
    "You generate short, concise task titles.\n\n"
    "Rules:\n"
    "- Every title MUST focus on the target object / main subject of the task.\n"
    "- Maximum 6 words\n"
    "- Noun phrase only — no action verbs\n"
    "- No media words (video/photo/image/clip/picture/footage)\n"
    "- No location names\n"
    "- No instructions (review/audit/validate/inspect/annotate/label)\n"
    "- Include time-of-day ONLY if it is critical\n\n"
    "Examples:\n"
    f"{_TITLE_FEW_SHOT_BLOCK}"
)

COT_SUFFIX = (
    "\n\nThink step by step:\n"
    "1. Identify the target object — the main subject being captured or described.\n"
    "2. Note any critical attribute (time-of-day, state, condition).\n"
    "3. Drop all verbs, media words, and location names.\n"
    "4. Compose a noun phrase of at most 6 words centered on the target object.\n"
    "Title:"
)


def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return

    if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError(
            "Local Title_gen requires torch and transformers. "
            "Use use_openrouter_titles=true to generate titles with OpenRouter instead."
        )

    if not _CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Title_gen checkpoint not found: {_CHECKPOINT_DIR}")

    dtype = torch.bfloat16 if _DEVICE.type == "cuda" else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(str(_CHECKPOINT_DIR))
    _model = AutoModelForSeq2SeqLM.from_pretrained(str(_CHECKPOINT_DIR), torch_dtype=dtype).to(_DEVICE)
    _model.eval()


def _infer(prompt: str, num_sequences: int = 1) -> List[str]:
    _load_model()
    enc = _tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(_DEVICE)

    num_sequences = max(1, min(num_sequences, 4))
    num_beams = max(4, num_sequences)

    with torch.no_grad():
        out_ids = _model.generate(
            **enc,
            max_new_tokens=16,
            num_beams=num_beams,
            num_return_sequences=num_sequences,
            early_stopping=True,
        )

    titles: List[str] = []
    for row in out_ids:
        titles.append(_tokenizer.decode(row, skip_special_tokens=True).strip())
    return titles


def _build_prompt(description: str, mode: TitleMode, target_object: str = "") -> str:
    text = description.strip()
    target_block = ""
    if target_object.strip():
        target_block = f"\nTarget object (every title must focus on this): {target_object.strip()}\n"
    if mode == "zero_shot":
        return f"Generate a short title focused on the target object for:{target_block}\n{text}\nTitle:"
    return f"{INSTRUCTION}{target_block}\nDescription: {text}{COT_SUFFIX}"


def clean_title(title: str) -> str:
    title = re.sub(r"^[\s\-•*\d.)]+", "", title or "")
    title = re.sub(r"\s+", " ", title).strip(" -:\n\t\"'")
    return " ".join(title.split()[:6])


def _title_matches_source_language(title: str, source_text: str) -> bool:
    source_arabic = _is_arabic_text(source_text)
    title_arabic = _is_arabic_text(title)
    return title_arabic if source_arabic else not title_arabic


def _title_language_rule(source_text: str) -> str:
    if _is_arabic_text(source_text):
        return (
            "The requester description is Arabic. Generate Arabic titles only. "
            "Do not use English or mix languages."
        )
    return (
        "The requester description is English. Generate English titles only. "
        "Do not use Arabic or mix languages."
    )


def _finalize_titles(raw_titles: List[str], n: int, source_text: str = "") -> List[str]:
    banned = re.compile(
        r"\b(photo|photos|image|images|video|videos|clip|picture|pictures|footage|task|"
        r"capture|send|submit|upload|record|film|take)\b",
        re.IGNORECASE,
    )
    unique: List[str] = []
    seen = set()

    for raw in raw_titles:
        title = clean_title(raw)
        if not title or banned.search(title):
            continue
        if source_text and not _title_matches_source_language(title, source_text):
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(title)

    return unique[:n]


def generate_titles_local(
    description: str,
    *,
    n: int = 3,
    mode: TitleMode = "few_shot_cot",
    target_object: str = "",
    language_source: str = "",
) -> List[str]:
    if not description.strip():
        raise ValueError("description is required.")

    prompt = _build_prompt(description, mode, target_object)
    raw = _infer(prompt, num_sequences=min(n, 4))

    if len(raw) < n and mode != "zero_shot":
        raw.extend(_infer(_build_prompt(description, "zero_shot", target_object), num_sequences=1))

    return _finalize_titles(raw, n=n, source_text=language_source or description)


def target_object_for_titles(extractions: Optional[Dict[str, Optional[str]]]) -> str:
    if not extractions:
        return ""
    value = extractions.get("target_object")
    return clean_text(str(value)) if value else ""


def model_status() -> dict:
    return {
        "checkpoint": str(_CHECKPOINT_DIR),
        "checkpoint_exists": _CHECKPOINT_DIR.is_dir(),
        "loaded": _model is not None,
        "device": str(_DEVICE),
    }

# ----- titles.py -----





class TitleGenerator:
    FEW_SHOT_EXAMPLES = TITLE_FEW_SHOT_EXAMPLES

    def __init__(self, backend: OpenRouterBackend):
        self.backend = backend

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"^[\s\-•*\d.)]+", "", title or "")
        title = re.sub(r"\s+", " ", title).strip(" -:\n\t\"'")
        words = title.split()
        return " ".join(words[:6])

    def _parse_titles(self, raw: str, n: int, language_source: str = "") -> List[str]:
        obj = try_parse_json(raw)
        candidates: List[str] = []

        if isinstance(obj, dict):
            raw_titles = obj.get("titles", [])
            if isinstance(raw_titles, list):
                candidates.extend(str(x) for x in raw_titles)

        if not candidates:
            for line in (raw or "").splitlines():
                cleaned = self._clean_title(line)
                if cleaned:
                    candidates.append(cleaned)

        unique: List[str] = []
        seen = set()
        banned = re.compile(
            r"\b(photo|photos|image|images|video|videos|clip|picture|pictures|footage|task|"
            r"capture|send|submit|upload|record|film|take)\b|"
            r"^task\s+title\s+suggestion\s+\d+$",
            re.IGNORECASE,
        )

        for title in candidates:
            title = self._clean_title(title)
            if not title or banned.search(title):
                continue
            if language_source and not _title_matches_source_language(title, language_source):
                continue
            key = title.lower()
            if key not in seen:
                seen.add(key)
                unique.append(title)

        return unique[:n]

    def generate(
        self,
        description: str,
        n: int = 3,
        target_object: str = "",
        language_source: str = "",
    ) -> List[str]:
        language_source = language_source or description
        few_shot_block = "\n".join([f"Description: {d}\nTitle: {t}" for d, t in self.FEW_SHOT_EXAMPLES])
        target_block = ""
        if target_object.strip():
            target_block = f"\nTarget object (every title MUST focus on this): {target_object.strip()}\n"
        system_prompt = f"""You generate short, concise crowdsourcing task titles.
Return ONLY valid JSON with this schema:
{{"titles": ["Title One", "Title Two", "Title Three"]}}
Rules:
- Generate exactly 3 distinct titles.
- Every title MUST focus on the target object / main subject — not the action, media type, or location.
- {_title_language_rule(language_source)}
- Maximum 6 words each.
- Noun phrase only.
- No action verbs such as capture, send, submit, upload, record, take, ارفع, التقط, صور, وثق, سجل.
- No media words such as photo, image, picture, video, clip, footage, صورة, صور, فيديو, مقطع.
- Keep titles professional and safe.
"""
        user_prompt = f"""Examples:
{few_shot_block}
{target_block}
Enhanced task description:
{description.strip()}

Return JSON only."""
        raw = self.backend.chat(system_prompt=system_prompt, user_message=user_prompt, max_tokens=220)
        return self._parse_titles(raw, n=n, language_source=language_source)


def generate_titles(
    description: str,
    backend: OpenRouterBackend,
    n: int = 3,
    target_object: str = "",
    language_source: str = "",
) -> Dict[str, Any]:
    titles = TitleGenerator(backend).generate(
        description,
        n=n,
        target_object=target_object,
        language_source=language_source or description,
    )
    return {"titles": titles, "selected_title": titles[0] if titles else ""}

# ----- DRF parsers + runners -----
_backend_cache: Dict[str, OpenRouterBackend] = {}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int(
    value: Any,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def parse_description(data: Any) -> str:
    description = (
        data.get("description")
        or data.get("text")
        or data.get("task_description")
        or ""
    )
    description = str(description).strip()
    if not description:
        raise ValueError("description is required.")
    return description


def parse_n_titles(data: Any, default: int = 3) -> int:
    return parse_int(data.get("n_titles") or data.get("n"), default=default, min_value=1, max_value=10)


def parse_use_openrouter_titles(data: Any) -> bool:
    return parse_bool(data.get("use_openrouter_titles"), default=False)


TitleMode = Literal["zero_shot", "few_shot_cot"]


def parse_title_mode(data: Any, default: TitleMode = "few_shot_cot") -> TitleMode:
    raw = data.get("title_mode") or data.get("mode") or data.get("generation_mode") or default
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in ("zero_shot", "zeroshot", "zero"):
        return "zero_shot"
    if mode in ("few_shot_cot", "few_shot", "cot", "fewshot"):
        return "few_shot_cot"
    raise ValueError("title_mode must be 'zero_shot' or 'few_shot_cot'.")


def _get_backend() -> OpenRouterBackend:
    primary = _configured_openrouter_model()
    fallbacks = _configured_openrouter_fallback_models()
    keys_file = str(_configured_openrouter_keys_file())
    cache_key = f"{primary}|{'|'.join(fallbacks)}|{keys_file}|0.0"

    if cache_key not in _backend_cache:
        _backend_cache[cache_key] = OpenRouterBackend(temperature=0.0)
    return _backend_cache[cache_key]


def _require_min_words(description: str) -> None:
    if len(meaningful_words(description)) < MIN_MEANINGFUL_WORDS:
        raise ValueError(f"Description too short (need at least {MIN_MEANINGFUL_WORDS} meaningful words).")


def run_health() -> Dict[str, Any]:
    try:
        model = _configured_openrouter_model()
        fallback_models = _configured_openrouter_fallback_models()
        keys_file = str(_configured_openrouter_keys_file())
        key_count = len(get_openrouter_api_keys())
        configured = key_count > 0
        error = None
    except RuntimeError as exc:
        model = ""
        fallback_models = []
        keys_file = str(getattr(settings, "OPENROUTER_KEYS_FILE", "") or "")
        key_count = 0
        configured = False
        error = str(exc)

    result = {
        "ok": configured,
        "model": model,
        "fallback_models": fallback_models,
        "openrouter_configured": configured,
        "openrouter_key_count": key_count,
        "openrouter_keys_file": keys_file,
    }
    if error:
        result["error"] = error
    return result


def run_title_model_health() -> Dict[str, Any]:
    status = model_status()
    return {"ok": status["checkpoint_exists"], "title_backend": "Title_gen", **status}


def run_extract(description: str) -> Dict[str, Any]:
    """
    Stage 1: safety check.
    Stage 2: visual/textual request-type check (only if safe).
    Stage 3: extraction + required-field check (target_object, action_required).
    """
    _require_min_words(description)
    backend = _get_backend()
    validation, raw, extractions = validate_before_work(description, backend)
    safety_check = validation.get("safety_check") or {}
    request_type_check = validation.get("request_type_check") or {}
    target_object_check = validation.get("target_object_check") or _empty_target_object_check()
    request_kind = _normalize_request_kind(request_type_check.get("request_type"))
    required_fields = validation.get("required_fields") or {}
    readiness = validation.get("readiness") or {}
    safety_ok = _safety_is_ok(safety_check)
    required_ok = required_fields.get("status") == "COMPLETE"
    target_object_ok = target_object_check.get("is_specific") is not False
    ready_for_process = bool(safety_ok and required_ok and target_object_ok)

    missing_labels: List[str] = []
    missing_questions: Dict[str, str] = {}
    extraction_parsed_ok = False
    if safety_ok and raw:
        _, extraction_parsed_ok = parse_llm_output(raw)
        summary = extraction_summary(extractions, raw, extraction_parsed_ok, request_kind)
        missing_labels = list(summary.get("missing_labels") or [])
        missing_questions = dict(summary.get("missing_questions") or {})

    if not safety_ok:
        recommended_questions = [UNSAFE_FOR_WORK_RECOMMENDED_MESSAGE]
    else:
        recommended_questions = _build_recommended_questions(
            missing_labels=missing_labels,
            missing_questions=missing_questions,
            readiness=readiness,
            missing_required=required_fields.get("missing_required_labels") or [],
            safety_ok=True,
            request_kind=request_kind,
            target_object_check=target_object_check,
        )

    media_autofill = _empty_media_autofill()
    media_autofill_parsed_ok = False
    media_autofill_error = None
    if safety_ok:
        try:
            media_autofill, _raw_media_autofill, media_autofill_parsed_ok = extract_media_autofill_fields(
                description,
                backend,
            )
        except Exception as exc:
            media_autofill_error = str(exc)

    extraction_payload = validation.get("extraction")
    if isinstance(extraction_payload, dict):
        extraction_payload = dict(extraction_payload)
        extraction_payload["media_autofill"] = media_autofill

    return {
        "ok": ready_for_process,
        "ready_for_process": ready_for_process,
        "safety_ok": safety_ok,
        "required_fields_ok": required_ok,
        "target_object_ok": target_object_ok,
        "extraction_parsed_ok": extraction_parsed_ok,
        "safety_check": safety_check,
        "request_type_check": request_type_check,
        "target_object_check": target_object_check,
        "extraction": extraction_payload,
        "required_fields": required_fields,
        "readiness": readiness,
        "extractions": extractions,
        "raw_extraction": raw,
        "raw_model_output": raw,
        "missing_labels": missing_labels,
        "missing_questions": missing_questions,
        "recommended_questions": recommended_questions,
        "media_autofill": media_autofill,
        "media_autofill_parsed_ok": media_autofill_parsed_ok,
        "media_autofill_error": media_autofill_error,
        "model": backend.model_name,
        "provider_used": backend.provider,
        "openrouter_key_count": len(backend.api_keys),
        "error": None
        if ready_for_process
        else _extract_blocker_message(safety_ok, required_fields, target_object_check),
    }


def _build_recommended_questions(
    *,
    missing_labels: List[str],
    missing_questions: Dict[str, str],
    readiness: Dict[str, Any],
    missing_required: List[str],
    safety_ok: bool,
    request_kind: Optional[str] = REQUEST_KIND_VISUAL,
    target_object_check: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Follow-up questions for safe text; single rewrite prompt when unsafe."""
    if not safety_ok:
        return [UNSAFE_FOR_WORK_RECOMMENDED_MESSAGE]

    questions = _questions_for_request_kind(request_kind)
    out: List[str] = []
    for label in missing_labels:
        q = missing_questions.get(label) or questions.get(label)
        if q:
            out.append(q)
    for label in missing_required:
        q = missing_questions.get(label) or questions.get(label)
        if q:
            out.append(q)
    target_check = target_object_check or {}
    if target_check.get("is_specific") is False:
        vague_question = clean_text(str(target_check.get("recommended_question") or ""))
        if vague_question:
            out.append(vague_question)
    for q in readiness.get("suggestions") or []:
        if q:
            out.append(str(q))
    return dedupe(out)


def _extract_blocker_message(
    safety_ok: bool,
    required_fields: Dict[str, Any],
    target_object_check: Optional[Dict[str, Any]] = None,
) -> str:
    if not safety_ok:
        return "Safety check failed. Call /process only after extract returns ready_for_process=true."
    target_check = target_object_check or {}
    if target_check.get("is_specific") is False:
        reason = clean_text(str(target_check.get("reason") or ""))
        if reason:
            return reason
        return "Target object is too vague. Please name a concrete subject."
    missing = required_fields.get("missing_required_labels") or []
    if missing:
        return "Missing required fields: " + ", ".join(missing) + "."
    return "Not ready for process."


def run_validate(description: str) -> Dict[str, Any]:
    _require_min_words(description)
    backend = _get_backend()
    validation, raw, extractions = validate_before_work(description, backend)
    can_enhance = bool(validation.get("can_enhance"))
    return {
        "ok": can_enhance,
        "can_continue_pipeline": can_enhance,
        "can_enhance": can_enhance,
        "safety_check": validation.get("safety_check"),
        "request_type_check": validation.get("request_type_check"),
        "extraction": validation.get("extraction"),
        "required_fields": validation.get("required_fields"),
        "validation": validation,
        "extractions": extractions,
        "raw_extraction": raw,
        "polished_description": "",
        "model": backend.model_name,
    }


def run_enhance(description: str) -> Dict[str, Any]:
    _require_min_words(description)
    backend = _get_backend()
    validation, raw, extractions = validate_before_work(description, backend)
    request_type_check = validation.get("request_type_check") or {}
    request_kind = _normalize_request_kind(request_type_check.get("request_type"))
    if not validation.get("can_enhance"):
        return {
            "ok": False,
            "error": "Enhancement blocked. Complete safety check and required fields first.",
            "safety_check": validation.get("safety_check"),
            "request_type_check": request_type_check,
            "extraction": validation.get("extraction"),
            "required_fields": validation.get("required_fields"),
            "validation": validation,
            "extractions": extractions,
            "raw_extraction": raw,
            "polished_description": "",
            "requester_facing_description": "",
            "requester_facing_skipped": True,
            "requester_facing_error": None,
            "enhancement": {},
            "model": backend.model_name,
        }
    polished, debug = run_enhancement_pipeline(
        description,
        backend,
        extractions=extractions,
        raw_original=raw,
        request_kind=request_kind,
    )
    debug["model"] = backend.model_name
    return {
        "ok": bool(polished),
        "safety_check": validation.get("safety_check"),
        "request_type_check": request_type_check,
        "extraction": validation.get("extraction"),
        "required_fields": validation.get("required_fields"),
        "validation": validation,
        "extractions": extractions,
        "raw_extraction": raw,
        "polished_description": polished,
        "requester_facing_description": debug.get("requester_facing_description", ""),
        "requester_facing_skipped": debug.get("requester_facing_skipped", False),
        "requester_facing_error": debug.get("requester_facing_error"),
        "enhancement": debug,
        "error": debug.get("error") if not polished else None,
        "model": backend.model_name,
    }


def run_titles(
    description: str,
    data: Any,
    *,
    target_object: str = "",
) -> Dict[str, Any]:
    n = parse_n_titles(data)
    if parse_use_openrouter_titles(data):
        backend = _get_backend()
        result = generate_titles(description, backend, n=n, target_object=target_object)
        result["title_backend"] = "openrouter"
        result["model"] = backend.model_name
        return result
    mode = parse_title_mode(data)
    titles = generate_titles_local(description, n=n, mode=mode, target_object=target_object)
    status = model_status()
    return {
        "ok": True,
        "titles": titles,
        "selected_title": titles[0] if titles else "",
        "title_backend": "Title_gen",
        "title_mode": mode,
        "model": status["checkpoint"],
        "device": status["device"],
    }


def run_titles_local_only(
    description: str,
    data: Any,
    *,
    n: Optional[int] = None,
    target_object: str = "",
    language_source: str = "",
) -> Dict[str, Any]:
    """Title suggestions via local Title_gen only."""
    n = n if n is not None else parse_n_titles(data)
    mode = parse_title_mode(data)
    titles = generate_titles_local(
        description,
        n=n,
        mode=mode,
        target_object=target_object,
        language_source=language_source or description,
    )
    status = model_status()
    return {
        "ok": True,
        "titles": titles,
        "selected_title": titles[0] if titles else "",
        "title_backend": "Title_gen",
        "title_mode": mode,
        "model": status["checkpoint"],
        "device": status["device"],
    }


def _unique_clean_titles(values: List[Any], n: int, source_text: str = "") -> List[str]:
    unique: List[str] = []
    seen = set()
    placeholder = re.compile(r"^task\s+title\s+suggestion\s+\d+$", re.IGNORECASE)
    for value in values:
        title = clean_text(str(value or ""))
        if not title or placeholder.search(title):
            continue
        if source_text and not _title_matches_source_language(title, source_text):
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(title)
        if len(unique) >= n:
            break
    return unique


def _merge_title_results(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    n: int,
    source_text: str = "",
) -> List[str]:
    values: List[Any] = []
    if isinstance(primary.get("titles"), list):
        values.extend(primary.get("titles") or [])
    if primary.get("selected_title"):
        values.append(primary.get("selected_title"))
    if isinstance(fallback.get("titles"), list):
        values.extend(fallback.get("titles") or [])
    if fallback.get("selected_title"):
        values.append(fallback.get("selected_title"))
    return _unique_clean_titles(values, n, source_text=source_text)


def run_process(
    description: str,
    data: Any,
) -> Dict[str, Any]:
    """
    Polish description, then generate titles.

    Title_gen is kept for English/local generation. Arabic descriptions and
    empty local-title outputs fall back to OpenRouter title generation so the UI
    receives title suggestions when processing succeeds.
    """
    _require_min_words(description)
    backend = _get_backend()
    n_titles = parse_n_titles(data)

    validation, raw, extractions = validate_before_work(description, backend)
    request_type_check = validation.get("request_type_check") or {}
    request_kind = _normalize_request_kind(request_type_check.get("request_type"))
    if not validation.get("can_enhance"):
        return {
            "ok": False,
            "error": "Enhancement blocked. Complete safety check and required fields first.",
            "safety_check": validation.get("safety_check"),
            "request_type_check": request_type_check,
            "extraction": validation.get("extraction"),
            "required_fields": validation.get("required_fields"),
            "polished_description": "",
            "requester_facing_description": "",
            "requester_facing_skipped": True,
            "requester_facing_error": None,
            "titles": [],
            "selected_title": "",
            "title_backend": "",
            "title_error": None,
            "sam3_prompts": _empty_sam3_prompts("SKIPPED_PROCESS_BLOCKED"),
            "model": backend.model_name,
            "provider_used": backend.provider,
            "openrouter_key_count": len(backend.api_keys),
        }

    polished, enhancement_debug = run_enhancement_pipeline(
        description,
        backend,
        extractions=extractions,
        raw_original=raw,
        request_kind=request_kind,
    )
    if not polished:
        return {
            "ok": False,
            "error": enhancement_debug.get("error", "Enhancement failed."),
            "safety_check": validation.get("safety_check"),
            "request_type_check": request_type_check,
            "extraction": validation.get("extraction"),
            "required_fields": validation.get("required_fields"),
            "polished_description": "",
            "requester_facing_description": "",
            "requester_facing_skipped": True,
            "requester_facing_error": None,
            "titles": [],
            "selected_title": "",
            "title_backend": "",
            "title_error": None,
            "sam3_prompts": _empty_sam3_prompts("SKIPPED_ENHANCEMENT_FAILED"),
            "model": backend.model_name,
            "provider_used": backend.provider,
            "openrouter_key_count": len(backend.api_keys),
        }

    title_backend = "Title_gen"
    title_error = None
    primary_title_result: Dict[str, Any] = {}
    fallback_title_result: Dict[str, Any] = {}
    target_object = target_object_for_titles(extractions)

    source_is_arabic = _is_arabic_text(description)
    use_openrouter_first = parse_use_openrouter_titles(data) or source_is_arabic

    if use_openrouter_first:
        title_backend = "openrouter"
        try:
            primary_title_result = generate_titles(
                polished,
                backend,
                n=n_titles,
                target_object=target_object,
                language_source=description,
            )
        except Exception as exc:
            title_error = str(exc)
            primary_title_result = {}
    else:
        try:
            primary_title_result = run_titles_local_only(
                polished,
                data,
                n=n_titles,
                target_object=target_object,
                language_source=description,
            )
        except Exception as exc:
            title_error = str(exc)
            primary_title_result = {}

    titles = _merge_title_results(
        primary_title_result, {}, n_titles, source_text=description
    )

    if len(titles) < n_titles:
        try:
            if use_openrouter_first:
                fallback_title_result = run_titles_local_only(
                    polished,
                    data,
                    n=n_titles,
                    target_object=target_object,
                    language_source=description,
                )
                fallback_backend = "Title_gen"
            else:
                fallback_title_result = generate_titles(
                    polished,
                    backend,
                    n=n_titles,
                    target_object=target_object,
                    language_source=description,
                )
                fallback_backend = "openrouter_fallback"
            titles = _merge_title_results(
                primary_title_result,
                fallback_title_result,
                n_titles,
                source_text=description,
            )
            if titles and title_backend != "openrouter":
                title_backend = f"Title_gen+{fallback_backend}"
            elif titles and title_backend == "openrouter" and fallback_title_result:
                title_backend = "openrouter+Title_gen_fallback"
        except Exception as exc:
            title_error = title_error or str(exc)

    selected_title = titles[0] if titles else ""
    try:
        sam3_prompts = build_sam3_prompts_for_process(
            description=description,
            extractions=extractions,
            request_kind=request_kind,
            backend=backend,
            title=selected_title,
        )
    except Exception as exc:
        sam3_prompts = _empty_sam3_prompts("SKIPPED_SAM3_ERROR", str(exc))

    return {
        "ok": True,
        "safety_check": validation.get("safety_check"),
        "request_type_check": request_type_check,
        "polished_description": polished,
        "requester_facing_description": enhancement_debug.get("requester_facing_description", ""),
        "requester_facing_skipped": enhancement_debug.get("requester_facing_skipped", False),
        "requester_facing_error": enhancement_debug.get("requester_facing_error"),
        "titles": titles,
        "selected_title": selected_title,
        "title_backend": title_backend,
        "title_error": title_error,
        "sam3_prompts": sam3_prompts,
        "model": backend.model_name,
        "provider_used": backend.provider,
        "openrouter_key_count": len(backend.api_keys),
    }