import logging
import re
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import requests
from django.conf import settings

from mobicrowd.views.apis.submission_flow.keyword_extraction import (
    convert_to_keywords_list,
    parse_structured_output,
    stability_check,
)
from mobicrowd.views.apis.submission_flow.qwen3_caption_relevance import (
    OPENROUTER_ENDPOINT,
    load_keys_from_file,
)

logger = logging.getLogger(__name__)


# ===============================
# CONFIGURATION
# ===============================

OPENROUTER_HTTP_REFERER = "http://localhost"
OPENROUTER_APP_TITLE = "Mobicrowd Task Understanding"

CROWDSOURCING_CONTEXT = """This is a visual content crowdsourcing system that handles both images and videos. The objective is to process textual task descriptions received from requesters and extract relevant information to determine the relevance of the visual content (photos and videos) submitted by workers to the requesters' requirements. The textual requests are also used to determine the skills and attributes associated with the workers required by the requesters to complete tasks involving visual media.
"""

PROMPT_TEMPLATE = """
{context}

You are an expert assistant tasked with extracting structured information from task descriptions for visual content crowdsourcing (images and videos).
Analyze the **Task Description** provided below and fill in the following template precisely.
While staying true to the task's core meaning, interpret the fields slightly broadly where appropriate to capture the full context.
Consider that tasks may involve capturing photos, recording videos, or both types of visual content.
**Do not** include the example or any explanation before the template.
Output **only** the filled template starting directly with `TASK_TOPIC:`.

**Template:**
TASK_TOPIC: [Main category or broader subject area of the visual content task]
ACTION_REQUIRED: [Primary action workers need to perform - capturing photos, recording videos, or both]
TARGET_OBJECT: [Key items, concepts, subjects, or scenes the visual content should focus on - OBJECTS/CONCEPTS ONLY. Do NOT include place names, city names, or location names here; those go in LOCATION_DETAILS. Be SPECIFIC - list exact objects, types, or categories (e.g. "street art, murals, graffiti" not "Melbourne street art, Melbourne murals"). Only include what is REQUIRED/REQUESTED. DO NOT include rejection language, exclusions, or restrictions.]
TARGET_DESCRIPTION: [Short, direct phrase for relevance matching: WHAT to capture (action + subject/object only). KEEP: object subtypes and categories that narrow the target (e.g. electrical, chemical, mechanical assets). KEEP restrictions/exclusions when present. REMOVE: quantities and counts (e.g. 10, 20, max 5), media types (photos, videos, images, clips), worker/provider requirements, equipment, camera specs, and location/place names. Example: "Send 10 photos of cats" → TARGET_DESCRIPTION: cats. Example: "Capture laboratory assets. They can be electrical, chemical, and mechanical assets. Provide 20 photos." → TARGET_DESCRIPTION: laboratory assets (electrical, chemical, and mechanical). Example: "Provide 15 photos of emergency exit signs. Exclude medical records." → TARGET_DESCRIPTION: emergency exit signs. Exclude medical records.]
LOCATION_DETAILS: [Specific place, area, or type of environment where visual content should be captured]
TIME_FRAME: [Deadline, urgency, or relevant time constraints for content capture]
WORKER_QUALIFICATIONS: [Mentioned skills, experience, ratings, or implicit attributes for visual content creation]
EQUIPMENT_NEEDED: [Specific tools, devices, cameras, or types of equipment necessary for visual capture]
COMPENSATION: [Payment amount, rate, or incentives mentioned]
VERIFICATION_METHOD: [How visual content completion will be confirmed or checked]
SAFETY_ISSUES: [Potential risks, hazards, or safety precautions mentioned for visual content capture]
PRIVACY_CONCERNS: [Data protection, anonymity, privacy considerations, or consent requirements for visual content]
RESTRICTIONS: [CRITICAL - MUST EXTRACT: List ALL EXCLUDED objects, types, or categories that should be REJECTED. Be VERY THOROUGH in detecting rejection phrases. Look for explicit words: "reject", "reject any", "reject all", "exclude", "exclude any", "no", "not", "avoid", "must not", "should not", "do not", "don't", "without", "except", "excluding", "prohibited", "forbidden", "banned". Also detect implicit rejections: "only X" means reject everything else. Examples: "reject any cats or dogs" → EXCLUDE: cats, dogs. "provide wildlife animals, reject cats or dogs" → EXCLUDE: cats, dogs. "only red cars" → EXCLUDE: all non-red cars. Format: "EXCLUDE: [comma-separated list of excluded items]. If no restrictions found, write "None". ALWAYS include this field - if you see ANY rejection language, extract it here.]
ADDITIONAL_CONDITIONS: [Other notable constraints, requirements, or context for visual content tasks]
RELEVANT_KEYWORDS: [Comma-separated list capturing main concepts, objects, actions, locations, visual elements, and context. Include **both specific terms from the description AND broader related themes or categories**.]

**Now, process the following task:**

Task Description: {statement}

**Filled Template:**
"""


# ===============================
# POST-PROCESSING FUNCTIONS
# ===============================

def clean_target_object(target_object: str) -> str:
    """Clean TARGET_OBJECT to remove restriction/rejection language."""
    if not target_object or target_object.lower() in ["none", "not specified", ""]:
        return target_object

    restriction_patterns = [
        r"that are not\s+[^,\.]+",
        r"excluding\s+[^,\.]+",
        r"except\s+[^,\.]+",
        r"without\s+[^,\.]+",
        r"other than\s+[^,\.]+",
        r"but not\s+[^,\.]+",
        r"no\s+[^,\.]+",
        r"reject\s+[^,\.]+",
        r"exclude\s+[^,\.]+",
        r"avoid\s+[^,\.]+",
        r"prohibited\s+[^,\.]+",
        r"forbidden\s+[^,\.]+",
    ]

    cleaned = target_object
    for pattern in restriction_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"[,\.]+$", "", cleaned).strip()
    return cleaned if cleaned else target_object


def remove_location_from_target_object(target_object: str) -> str:
    """
    Remove location/place name prefixes from TARGET_OBJECT so it contains only objects/concepts.
    E.g. "Melbourne street art, Melbourne murals, Melbourne graffiti" -> "street art, murals, graffiti".
    """
    if not target_object or not target_object.strip():
        return target_object
    if target_object.lower() in ["none", "not specified", ""]:
        return target_object

    parts = [p.strip() for p in target_object.split(",") if p.strip()]
    if not parts:
        return target_object

    def first_word(s: str) -> str:
        return s.split(None, 1)[0].strip() if s.strip() else ""

    multi_word = [p for p in parts if len(p.split()) > 1]
    if not multi_word:
        return target_object

    leading_counts: Dict[str, int] = {}
    for p in multi_word:
        w = first_word(p)
        if w:
            leading_counts[w] = leading_counts.get(w, 0) + 1

    prefix_to_strip = None
    for w, count in leading_counts.items():
        if count >= 2 and count >= len(multi_word) * 0.5:
            prefix_to_strip = w
            break

    if prefix_to_strip:
        normalized = []
        for p in parts:
            if p.startswith(prefix_to_strip + " ") or p == prefix_to_strip:
                rest = p[len(prefix_to_strip):].strip() if p != prefix_to_strip else ""
                if rest:
                    normalized.append(rest)
            else:
                normalized.append(p)
        seen = set()
        unique = [x for x in normalized if x.lower() not in seen and not seen.add(x.lower())]
        return ", ".join(unique)

    seen = set()
    unique = [p for p in parts if p.lower() not in seen and not seen.add(p.lower())]
    return ", ".join(unique)


_LOCATION_TERMS_IN_TARGET = frozenset(
    {"locations", "coordinates", "gps", "position", "positions", "geolocation", "geo-location", "map", "address", "place"}
)


def strip_location_terms_from_target_object(target_object: str) -> str:
    """Remove standalone location-related terms from TARGET_OBJECT (e.g. locations, coordinates, GPS)."""
    if not target_object or not target_object.strip():
        return target_object
    if target_object.lower() in ["none", "not specified", ""]:
        return target_object

    parts = [p.strip() for p in target_object.split(",") if p.strip()]
    filtered = [p for p in parts if p.lower() not in _LOCATION_TERMS_IN_TARGET]
    if not filtered:
        return target_object
    return ", ".join(filtered)


def clean_restrictions_format(restrictions: str) -> str:
    """
    Normalize RESTRICTIONS field to either:
      - "EXCLUDE: item1, item2, ..."
      - "None"
    """
    if not restrictions or restrictions.lower() in ["none", "not specified", ""]:
        return "None"

    excluded_items: List[str] = []

    exclude_pattern = r"EXCLUDE:\s*([^;]+?)(?=;|EXCLUDE:|REJECT:|REQUIRE_ONLY:|$)"
    for match in re.finditer(exclude_pattern, restrictions, re.IGNORECASE | re.DOTALL):
        items_str = match.group(1).strip()
        excluded_items.extend([item.strip() for item in items_str.split(",") if item.strip()])

    reject_pattern = r"REJECT:\s*([^;]+?)(?=;|EXCLUDE:|REJECT:|REQUIRE_ONLY:|$)"
    for match in re.finditer(reject_pattern, restrictions, re.IGNORECASE | re.DOTALL):
        items_str = match.group(1).strip()
        excluded_items.extend([item.strip() for item in items_str.split(",") if item.strip()])

    require_only_pattern = r"REQUIRE_ONLY:\s*([^;]+?)(?=;|EXCLUDE:|REJECT:|REQUIRE_ONLY:|$)"
    for match in re.finditer(require_only_pattern, restrictions, re.IGNORECASE | re.DOTALL):
        items_str = match.group(1).strip()
        excluded_items.extend([item.strip() for item in items_str.split(",") if item.strip()])

    if not excluded_items:
        free_text_patterns = [
            r"(?:reject|exclude|excluding)\s+(?:any|all)?\s*([^,\.;]+?)(?:\.|,|;|$)",
            r"no\s+([^,\.;]+?)(?:\.|,|;|$)",
            r"not\s+([^,\.;]+?)(?:\.|,|;|$)",
        ]
        for pattern in free_text_patterns:
            for match in re.finditer(pattern, restrictions, re.IGNORECASE):
                item = match.group(1).strip()
                if item and len(item) > 1:
                    excluded_items.append(item)

    cleaned_items: List[str] = []
    for item in excluded_items:
        item = re.sub(r"^only\s+", "", item, flags=re.IGNORECASE).strip()
        item = re.sub(r"[,\.;]+$", "", item).strip()
        item = re.sub(r"^(any|all)\s+", "", item, flags=re.IGNORECASE).strip()
        if item and len(item) > 1:
            cleaned_items.append(item)

    seen = set()
    unique_items = []
    for item in cleaned_items:
        il = item.lower()
        if il not in seen:
            seen.add(il)
            unique_items.append(item)

    return f"EXCLUDE: {', '.join(unique_items)}" if unique_items else "None"


def extract_restrictions_from_text(text: str) -> str:
    """Fallback restriction extraction if LLM leaves it empty."""
    if not text:
        return "None"

    text_lower = text.lower()
    excluded_items: List[str] = []

    reject_patterns = [
        r"reject\s+(?:any|all)?\s*([^,\.]+?)(?:\.|,|$)",
        r"reject\s+(?:any|all)?\s*([^,\.]+?)(?:\s+or\s+)([^,\.]+?)(?:\.|,|$)",
    ]
    for pattern in reject_patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            excluded_items.extend([g.strip() for g in match.groups() if g])

    exclude_patterns = [
        r"exclude\s+(?:any|all)?\s*([^,\.]+?)(?:\.|,|$)",
        r"excluding\s+([^,\.]+?)(?:\.|,|$)",
    ]
    for pattern in exclude_patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            excluded_items.extend([g.strip() for g in match.groups() if g])

    no_patterns = [
        r"(?:^|,|\s)no\s+([^,\.]+?)(?:\.|,|$)",
        r"(?:^|,|\s)not\s+([^,\.]+?)(?:\.|,|$)",
    ]
    for pattern in no_patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            excluded_items.extend([g.strip() for g in match.groups() if g])

    excluded_items = [item.strip() for item in excluded_items if item.strip()]
    excluded_items = list(dict.fromkeys(excluded_items))

    return f"EXCLUDE: {', '.join(excluded_items)}" if excluded_items else "None"


# -------------------------------
# TARGET_DESCRIPTION CLEANING (requested)
# -------------------------------

_MEDIA_WORDS = frozenset({
    "photo", "photos", "photograph", "photographs", "image", "images", "picture", "pictures",
    "video", "videos", "clip", "clips", "footage", "recording", "recordings"
})

_MEDIA_TO_NEUTRAL = {
    "photo": "content",
    "photos": "content",
    "photograph": "content",
    "photographs": "content",
    "image": "content",
    "images": "content",
    "picture": "content",
    "pictures": "content",
    "video": "content",
    "videos": "content",
    "clip": "content",
    "clips": "content",
    "footage": "content",
    "recording": "content",
    "recordings": "content",
}

_GENERIC_ENV_KEYWORDS = frozenset({
    "street", "road", "sidewalk", "crosswalk",
    "parking lot", "car park", "garage",
    "construction site", "worksite",
    "store", "shop", "supermarket", "mall", "retail",
    "office", "warehouse", "factory",
    "home", "house", "kitchen", "living room",
    "park", "playground",
    "beach", "coast",
    "forest", "woods", "trail",
    "river", "lake", "ocean", "sea",
    "stadium", "field", "court",
    "restaurant", "cafe",
})

_TIME_OF_DAY_PATTERNS = [
    (r"\b(at\s+night|during\s+the\s+night|nighttime)\b", "At night"),
    (r"\b(in\s+the\s+morning|early\s+morning)\b", "In the morning"),
    (r"\b(in\s+the\s+afternoon)\b", "In the afternoon"),
    (r"\b(in\s+the\s+evening)\b", "In the evening"),
    (r"\b(at\s+dawn|sunrise)\b", "At dawn"),
    (r"\b(at\s+sunset)\b", "At sunset"),
    (r"\b(midday|noon)\b", "At midday"),
]

_PRIVACY_PATTERNS = [
    (r"\bblur\s+faces?\b", "Blur faces"),
    (r"\bblur\s+license\s+plates?\b", "Blur license plates"),
    (r"\b(no|avoid|do\s+not\s+include)\s+(identifiable\s+)?faces?\b", "No identifiable faces"),
    (r"\b(no|avoid|do\s+not\s+include)\s+(identifiable\s+)?people\b", "No identifiable people"),
]

_QUALITY_PATTERNS = [
    (r"\b(no|not)\s+blurry\b", "No blurry content"),
    (r"\b(in\s+focus|well\s+focused)\b", "Keep content in focus"),
    (r"\b(text)\s+(is\s+)?readable\b", "Ensure text is readable"),
    (r"\b(good\s+lighting|well\s+lit)\b", "Use good lighting"),
]


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _looks_unintelligible(text: str) -> bool:
    """
    Heuristic: if the text looks like junk (random chars, too few words for its length),
    do not rewrite it.
    """
    t = _normalize_ws(text)
    if not t:
        return True
    # very short strings are typically fine (not unintelligible)
    if len(t) <= 18:
        return False

    letters = re.findall(r"[A-Za-z]", t)
    if not letters:
        return True

    alpha_ratio = len(letters) / max(1, len(t))
    if alpha_ratio < 0.45:
        return True

    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(t) > 60 and len(words) < 4:
        return True

    return False


def _strip_quantities_and_media(text: str) -> str:
    """
    Remove quantities/durations/per-contributor limits and neutralize media words.
    TARGET_DESCRIPTION should not contain:
      - "50 images", "max 5 photos", "10-20 seconds", etc.
      - explicit media type words (photo/video/image)
    """
    s = " " + _normalize_ws(text) + " "

    # remove durations like 8-12s, 10 seconds, 10-20 seconds, etc.
    s = re.sub(r"\b\d+\s*(?:-\s*\d+)?\s*(?:seconds|second|secs|sec|s)\b", " ", s, flags=re.IGNORECASE)

    # remove quantity constraints involving media nouns (repeat for chained counts)
    media_noun = r"photos?|pictures?|images?|videos?|clips?|recordings?|shots?"
    for _ in range(4):
        s = re.sub(
            rf"\b(max(?:imum)?|min(?:imum)?|at\s+least|no\s+more\s+than|up\s+to)\s+\d+\b.*?\b({media_noun})\b",
            " ",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            rf"\b\d+\s*({media_noun})\b(\s+and\s+)?",
            " ",
            s,
            flags=re.IGNORECASE,
        )

    # remove "send/submit N of" and bare leading counts
    s = re.sub(
        r"\b(send|submit|provide|upload|deliver|capture)\s+(\d+\s+)?(of\s+)?",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\b(at\s+least|up\s+to|max(?:imum)?|min(?:imum)?)\s+\d+\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\b", " ", s)

    # neutralize media words (keeps grammar readable)
    def repl(m):
        w = m.group(0).lower()
        return _MEDIA_TO_NEUTRAL.get(w, "")

    s = re.sub(r"\b(" + "|".join(sorted(_MEDIA_WORDS, key=len, reverse=True)) + r")\b", repl, s, flags=re.IGNORECASE)

    # remove common submission/format phrasing
    s = re.sub(r"\b(upload|submit|provide|deliver)\b\s+(the\s+)?(content|files?)\b", " ", s, flags=re.IGNORECASE)

    return _normalize_ws(s)


def clean_target_description_for_relevance(text: str) -> str:
    """
    Normalize TARGET_DESCRIPTION for relevance/keyword matching:
    direct subject/action only — no counts, media types, or submission phrasing.
    """
    s = _normalize_ws(text)
    if not s:
        return s

    s = _strip_quantities_and_media(s)
    s = re.sub(r"\b(and|or)\s+(of|and|or)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcapture\s+of\b", "capture", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcapture\s+and\s+", "capture ", s, flags=re.IGNORECASE)
    s = re.sub(r"^(of|about)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(send|submit|provide|upload|deliver)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(capture|photograph|record|document)\s+", "", s, flags=re.IGNORECASE)
    s = _normalize_ws(s)

    if not s:
        return _normalize_ws(text)

    # Prefer a simple direct phrase; avoid empty "Capture ."
    if not re.match(r"^(capture|photograph|record|document)\b", s, flags=re.IGNORECASE):
        s = f"Capture {s}"

    return _normalize_ws(s.rstrip(" ."))


_SUBTYPE_CLAUSE_PATTERNS = [
    r"(?:they|it|these|those)\s+can\s+be\s+(.+?)(?:\.|;|$)",
    r"(?:including|such\s+as|like)\s+(.+?)(?:\.|;|$)",
]


def _extract_subtype_qualifiers(original_text: str) -> str:
    """Extract object subtype lists from the original description."""
    orig = _normalize_ws(original_text)
    if not orig:
        return ""

    for pattern in _SUBTYPE_CLAUSE_PATTERNS:
        match = re.search(pattern, orig, flags=re.IGNORECASE)
        if not match:
            continue
        clause = _normalize_ws(match.group(1))
        clause = _strip_quantities_and_media(clause)
        clause = re.sub(r"^(the|a|an)\s+", "", clause, flags=re.IGNORECASE)
        clause = re.sub(r"\s+assets?\s*$", "", clause, flags=re.IGNORECASE).strip()
        if clause and len(clause) >= 3:
            return clause

    return ""


def enrich_target_description_from_original(original_text: str, target_description: str) -> str:
    """Re-attach useful subtype qualifiers when the LLM summary drops them."""
    td = _normalize_ws(target_description)
    if not td:
        return td

    qualifiers = _extract_subtype_qualifiers(original_text)
    if not qualifiers:
        return td

    td_lower = td.lower()
    qual_lower = qualifiers.lower()
    if qual_lower in td_lower:
        return td

    parts = re.split(r",|\band\b", qualifiers, flags=re.IGNORECASE)
    significant = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    if significant and all(p.lower() in td_lower for p in significant):
        return td

    base = re.sub(r"\s*\([^)]+\)\s*$", "", td).strip()
    return f"{base} ({qualifiers})"


def _extract_first_match(text: str, patterns) -> str:
    tl = text.lower()
    for pat, out in patterns:
        if re.search(pat, tl, flags=re.IGNORECASE):
            return out
    return ""


def _is_specific_place_name(location_details: str) -> bool:
    """
    Very rough heuristic to detect that LOCATION_DETAILS likely contains a specific place name.
    If it looks specific, we won't include it in TARGET_DESCRIPTION.
    """
    loc = _normalize_ws(location_details)
    if not loc:
        return False

    # commas often indicate specific address/city/country formatting
    if "," in loc:
        return True

    # contains a country/state-style token
    if re.search(r"\b(usa|u\.s\.a\.|united\s+states|tunisia|france|germany|italy|uk|u\.k\.)\b", loc, re.IGNORECASE):
        return True

    # multiple capitalized words suggests a named place
    caps = re.findall(r"\b[A-Z][a-z]{2,}\b", loc)
    if len(caps) >= 2:
        return True

    return False


def _maybe_include_generic_environment(location_details: str) -> str:
    """
    Include only GENERIC environments (parking lot, street, construction site, etc.),
    never specific city/country names.
    """
    loc = _normalize_ws(location_details)
    if not loc:
        return ""
    if _is_specific_place_name(loc):
        return ""

    loc_l = loc.lower()
    # allow generic environment keywords
    for kw in _GENERIC_ENV_KEYWORDS:
        if kw in loc_l:
            # keep original location phrase if it already starts with a preposition
            if re.match(r"^(in|at|on|near|inside|outside)\b", loc_l):
                return loc
            return f"in {loc}"
    return ""


# If original description is this short (chars), keep it as-is — do not generate or use hardcoded text.
_SHORT_DESCRIPTION_THRESHOLD = 30


def build_clean_target_description(original_text: str, structured_output: Dict[str, Any]) -> str:
    """
    Build a clean TARGET_DESCRIPTION:
      - Provide + main target objects (concise)
      - Add essential acceptance constraints (privacy/quality/time-of-day) if present in original
      - Add exclusions (RESTRICTIONS) if present
    If input is short, unintelligible, or we can't identify targets, return original unchanged.
    """
    orig = _normalize_ws(original_text)
    if not orig:
        return orig
    # Short description: keep as-is, do not generate or use hardcoded text
    if len(orig) <= _SHORT_DESCRIPTION_THRESHOLD:
        return orig
    if _looks_unintelligible(orig):
        return orig

    target_object = _normalize_ws(str(structured_output.get("TARGET_OBJECT", "") or ""))
    restrictions = _normalize_ws(str(structured_output.get("RESTRICTIONS", "") or ""))
    location_details = _normalize_ws(str(structured_output.get("LOCATION_DETAILS", "") or ""))
    additional = _normalize_ws(str(structured_output.get("ADDITIONAL_CONDITIONS", "") or ""))
    privacy_field = _normalize_ws(str(structured_output.get("PRIVACY_CONCERNS", "") or ""))

    # If we have no meaningful target object, do not rewrite.
    if not target_object or target_object.lower() in ["none", "not specified"]:
        return orig

    # Choose up to 3 main target terms (prefer those that appear in the original text)
    terms = [t.strip() for t in target_object.split(",") if t.strip()]
    if not terms:
        return orig

    orig_l = orig.lower()
    in_orig = []
    for t in terms:
        tl = t.lower()
        # simple substring match (robust for multiword phrases)
        if tl in orig_l:
            in_orig.append(t)

    chosen = (in_orig[:3] if in_orig else terms[:3])
    target_core = ", ".join(chosen).strip()
    if not target_core:
        return orig

    # Always use a generic verb (no media type)
    base = f"Capture {target_core}"

    env = _maybe_include_generic_environment(location_details)
    if env:
        base = f"{base} {env}"

    extras: List[str] = []

    # time-of-day (from original only)
    tod = _extract_first_match(orig, _TIME_OF_DAY_PATTERNS)
    if tod:
        extras.append(tod)

    # privacy constraints (from original OR extracted privacy field)
    privacy_text = orig
    if privacy_field:
        privacy_text = privacy_text + " " + privacy_field
    priv = _extract_first_match(privacy_text, _PRIVACY_PATTERNS)
    if priv:
        extras.append(priv)

    # quality constraints (from original OR additional)
    quality_text = orig + (" " + additional if additional else "")
    qual = _extract_first_match(quality_text, _QUALITY_PATTERNS)
    if qual:
        extras.append(qual)

    # exclusions
    if restrictions and restrictions.lower() not in ["none", "not specified"]:
        extras.append(restrictions)

    cleaned = base
    if extras:
        cleaned = cleaned.rstrip(".") + ". " + ". ".join([e.rstrip(".") for e in extras if e])

    cleaned = _strip_quantities_and_media(cleaned)
    cleaned = clean_target_description_for_relevance(cleaned)
    cleaned = enrich_target_description_from_original(orig, cleaned)

    # If cleaning made it useless, keep original
    if not cleaned or len(cleaned) < 8:
        return orig

    return cleaned


def post_process_extraction(result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """Post-process extraction results: clean TARGET_OBJECT, ensure RESTRICTIONS, and build clean TARGET_DESCRIPTION."""
    structured_output = result.get("structured_output", {}) or {}

    # Clean TARGET_OBJECT (remove restriction language + strip location-related pollution)
    if "TARGET_OBJECT" in structured_output:
        structured_output["TARGET_OBJECT"] = clean_target_object(structured_output["TARGET_OBJECT"])
        structured_output["TARGET_OBJECT"] = remove_location_from_target_object(structured_output["TARGET_OBJECT"])
        structured_output["TARGET_OBJECT"] = strip_location_terms_from_target_object(structured_output["TARGET_OBJECT"])

    # Normalize / ensure RESTRICTIONS
    restrictions = _normalize_ws(str(structured_output.get("RESTRICTIONS", "") or ""))
    if restrictions and restrictions.lower() not in ["none", "not specified"]:
        structured_output["RESTRICTIONS"] = clean_restrictions_format(restrictions)
    else:
        extracted = extract_restrictions_from_text(original_text)
        structured_output["RESTRICTIONS"] = extracted if extracted else "None"

    # Build CLEAN TARGET_DESCRIPTION (requested behavior)
    structured_output["TARGET_DESCRIPTION"] = build_clean_target_description(original_text, structured_output)

    result["structured_output"] = structured_output
    return result


def expand_target_object_with_synonyms(
    target_object: str,
    model: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
) -> str:
    """
    Expand TARGET_OBJECT with synonyms and related terms so matching is more exhaustive.
    Returns a comma-separated string: original terms plus synonyms and closely related concepts.
    """
    if not target_object or not target_object.strip():
        return target_object
    if target_object.lower() in ["none", "not specified", ""]:
        return target_object

    prompt = f"""You are helping a visual content matching system. Given the following TARGET_OBJECT(s) from a task description, expand the list to be more exhaustive for matching.

Original TARGET_OBJECT: {target_object}

Add:
- Synonyms (e.g. "car" → "automobile", "vehicle")
- Closely related terms (e.g. "firefighter" → "fireman", "fire crew")
- Common alternative names or categories (e.g. "wildlife" → "wild animals", "fauna")
- Same concept in different wording

Rules:
- Keep all original terms (objects/concepts only).
- Add only relevant synonyms and related terms; do not add unrelated concepts.
- Do NOT include place names, city names, or location names.
- Output a single comma-separated list, no numbering or bullets.
- Do not include exclusion/rejection language.
- Keep the list concise but exhaustive (typically 1.5x to 3x the original length).

Output only the expanded comma-separated list, nothing else:"""

    try:
        model_id = _resolve_primary_model(
            model,
            "OPENROUTER_TARGET_EXPANSION_MODEL",
        )
        model_fallbacks = _resolve_fallback_models(
            fallback_models,
            "OPENROUTER_TARGET_EXPANSION_FALLBACK_MODELS",
        )

        out = openrouter_chat_completion(
            messages=[
                {"role": "system", "content": "You output only comma-separated lists for expanding target objects. No explanations."},
                {"role": "user", "content": prompt},
            ],
            model=model_id,
            fallback_models=model_fallbacks,
            temperature=0.2,
            max_tokens=512,
        )
        out = (out or "").strip()
        if out:
            out = re.sub(r"\s*,\s*", ", ", out)
            out = re.sub(r"\s+", " ", out).strip()
            out = remove_location_from_target_object(out)
            out = strip_location_terms_from_target_object(out)
            return out
    except Exception as e:
        logger.warning("TARGET_OBJECT expansion failed (using original): %s", e)

    return target_object


# ===============================
# CORE FUNCTIONS
# ===============================
_OPENROUTER_SESSION: Optional[requests.Session] = None


def _resolve_primary_model(explicit_model: Optional[str], setting_name: str) -> str:
    """Resolve a role model from an explicit override or Django settings."""
    model = str(
        explicit_model
        if explicit_model is not None
        else getattr(settings, setting_name, "")
    ).strip()
    if not model:
        raise RuntimeError(f"{setting_name} is not configured.")
    return model


def _resolve_fallback_models(
    explicit_fallbacks: Optional[List[str]],
    setting_name: str,
) -> List[str]:
    """Resolve, normalize, and de-duplicate role fallback models."""
    values = (
        explicit_fallbacks
        if explicit_fallbacks is not None
        else getattr(settings, setting_name, [])
    )

    if values is None:
        values = []
    elif isinstance(values, str):
        values = [values]

    out: List[str] = []
    seen = set()
    for value in values:
        model = str(value or "").strip()
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def _build_model_chain(primary_model: str, fallback_models: Optional[List[str]]) -> List[str]:
    """Return primary + unique fallbacks in OpenRouter priority order."""
    chain: List[str] = []
    seen = set()
    for value in [primary_model, *(fallback_models or [])]:
        model = str(value or "").strip()
        if model and model not in seen:
            seen.add(model)
            chain.append(model)

    if not chain:
        raise RuntimeError("No OpenRouter model is configured.")
    return chain


def _get_openrouter_session() -> requests.Session:
    global _OPENROUTER_SESSION
    if _OPENROUTER_SESSION is None:
        _OPENROUTER_SESSION = requests.Session()
    return _OPENROUTER_SESSION


def _load_openrouter_keys() -> List[str]:
    """
    The file may still contain one or multiple keys, so the existing key-rotation
    behavior remains compatible even when production uses a single paid key.
    """
    keys_file = getattr(settings, "OPENROUTER_KEYS_FILE", None)
    if not keys_file:
        raise RuntimeError("OPENROUTER_KEYS_FILE is not configured in Django settings.")

    path = Path(str(keys_file))
    if not path.exists():
        raise RuntimeError(f"OPENROUTER_KEYS_FILE not found: {path}")

    keys = load_keys_from_file(str(path))
    if not keys:
        raise RuntimeError(f"No valid OpenRouter keys found in: {path}")

    return keys


def openrouter_chat_completion(
    *,
    messages: List[Dict[str, str]],
    model: str,
    fallback_models: Optional[List[str]] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    max_attempts: int = 6,
    timeout_s: float = 60.0,
) -> str:
    """
    Call OpenRouter with:
      1) provider fallback for each model (allow_fallbacks=True), and
      2) native cross-model fallback via the ordered `models` array.

    OpenRouter tries the primary model first, then each configured fallback if
    the previous model cannot serve the request (for example unavailable model,
    provider exhaustion/rate limiting, moderation refusal, or model/API error).

    A normal successful completion is returned immediately; application-level
    semantic outcomes are not treated as fallback conditions here.
    """
    keys = _load_openrouter_keys()
    session = _get_openrouter_session()
    model_chain = _build_model_chain(model, fallback_models)
    last_error: Any = None

    for attempt in range(max_attempts):
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

        logger.info(
            "OpenRouter task call attempt=%s/%s model_chain=%s",
            attempt + 1,
            max_attempts,
            model_chain,
        )

        try:
            response = session.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=(5.0, float(timeout_s)),
            )
        except requests.RequestException as exc:
            last_error = {
                "type": "request_exception",
                "message": str(exc),
            }
            logger.warning(
                "OpenRouter request failed attempt=%s/%s model_chain=%s err=%s",
                attempt + 1,
                max_attempts,
                model_chain,
                exc,
            )
            continue

        try:
            data = response.json()
        except ValueError:
            last_error = {
                "type": "invalid_json_response",
                "http_status": response.status_code,
                "body": (response.text or "")[:2000],
            }
            logger.warning(
                "OpenRouter returned invalid JSON attempt=%s/%s status=%s model_chain=%s",
                attempt + 1,
                max_attempts,
                response.status_code,
                model_chain,
            )
            continue

        if response.status_code == 200 and data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")
            logger.info(
                "OpenRouter task call success attempt=%s/%s requested_models=%s served_model=%s provider=%s",
                attempt + 1,
                max_attempts,
                model_chain,
                data.get("model"),
                data.get("provider"),
            )
            return (content or "").strip()

        last_error = {
            "type": "openrouter_error",
            "http_status": response.status_code,
            "response": data,
        }
        logger.warning(
            "OpenRouter chat failed attempt=%s/%s status=%s model_chain=%s error=%r",
            attempt + 1,
            max_attempts,
            response.status_code,
            model_chain,
            data.get("error") if isinstance(data, dict) else data,
        )

    raise RuntimeError(
        f"OpenRouter chat completion failed after {max_attempts} attempts "
        f"for model_chain={model_chain}: {last_error}"
    )


def extract_task_info_with_stability(
    task_description: str,
    model: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    tries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Tuple[str, float]:
    """Extract structured task info with stability checking (multiple attempts)."""
    prompt = PROMPT_TEMPLATE.format(context=CROWDSOURCING_CONTEXT, statement=task_description)
    outputs: List[str] = []

    model_id = _resolve_primary_model(
        model,
        "OPENROUTER_TASK_UNDERSTANDING_MODEL",
    )
    model_fallbacks = _resolve_fallback_models(
        fallback_models,
        "OPENROUTER_TASK_UNDERSTANDING_FALLBACK_MODELS",
    )

    for attempt in range(tries):
        try:
            result = openrouter_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert assistant for extracting structured information from task descriptions."},
                    {"role": "user", "content": prompt},
                ],
                model=model_id,
                fallback_models=model_fallbacks,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            match = re.search(r"TASK_TOPIC:.*", result.strip(), re.DOTALL | re.IGNORECASE)
            extracted = match.group(0).strip() if match else ""
            outputs.append(extracted)
        except Exception as e:
            logger.warning("Error in extraction attempt %s: %s", attempt + 1, e)
            outputs.append("")

    avg_stability = stability_check(outputs)
    return outputs[0] if outputs else "", avg_stability


def extract_detailed_task_info(
    text: str,
    model: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Detailed extraction via OpenRouter using the template."""
    prompt = PROMPT_TEMPLATE.format(context=CROWDSOURCING_CONTEXT, statement=text)

    model_id = _resolve_primary_model(
        model,
        "OPENROUTER_TASK_UNDERSTANDING_MODEL",
    )
    model_fallbacks = _resolve_fallback_models(
        fallback_models,
        "OPENROUTER_TASK_UNDERSTANDING_FALLBACK_MODELS",
    )

    default_result = {
        "TASK_TOPIC": "",
        "ACTION_REQUIRED": "",
        "TARGET_OBJECT": "",
        "TARGET_DESCRIPTION": "",
        "LOCATION_DETAILS": "",
        "TIME_FRAME": "",
        "WORKER_QUALIFICATIONS": "",
        "EQUIPMENT_NEEDED": "",
        "COMPENSATION": "",
        "VERIFICATION_METHOD": "",
        "SAFETY_ISSUES": "",
        "PRIVACY_CONCERNS": "",
        "RESTRICTIONS": "",
        "ADDITIONAL_CONDITIONS": "",
        "RELEVANT_KEYWORDS": ""
    }

    try:
        result_text = openrouter_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert assistant for extracting structured information from task descriptions for visual content crowdsourcing."},
                {"role": "user", "content": prompt},
            ],
            model=model_id,
            fallback_models=model_fallbacks,
            temperature=0.1,
            max_tokens=1024,
        )
        logger.debug("OpenRouter response (first 500 chars): %s", result_text[:500] if result_text else "")

        result = default_result.copy()

        pattern = r'^([A-Z_]+):\s*(.*?)(?=\n[A-Z_]+:|$)'
        matches = re.finditer(pattern, result_text, re.MULTILINE | re.DOTALL)
        for match in matches:
            key = match.group(1).strip()
            value = ' '.join(match.group(2).strip().split())
            if key in result:
                result[key] = value

        # Fallback line parsing if regex failed
        if not any(result.values()):
            lines = result_text.strip().split('\n')
            current_key = None
            current_value: List[str] = []
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith(' '):
                    if current_key and current_key in result:
                        result[current_key] = ' '.join(current_value).strip()
                    parts = line.split(':', 1)
                    current_key = parts[0].strip()
                    current_value = [parts[1].strip()] if len(parts) == 2 and parts[1].strip() else []
                elif current_key and current_key in result:
                    current_value.append(line)
            if current_key and current_key in result:
                result[current_key] = ' '.join(current_value).strip()

        # Ensure RESTRICTIONS not empty
        rv = result.get("RESTRICTIONS", "")
        if not rv or rv.lower() in ["none", "not specified", ""]:
            extracted = extract_restrictions_from_text(text)
            if extracted != "None":
                result["RESTRICTIONS"] = extracted

        if result.get("TARGET_DESCRIPTION"):
            cleaned_td = clean_target_description_for_relevance(result["TARGET_DESCRIPTION"])
            result["TARGET_DESCRIPTION"] = enrich_target_description_from_original(text, cleaned_td)

        # Basic clean on TARGET_OBJECT (more cleaning happens in post_process_extraction)
        if 'TARGET_OBJECT' in result:
            result['TARGET_OBJECT'] = clean_target_object(result['TARGET_OBJECT'])

        return result

    except Exception as e:
        logger.exception("OpenRouter detailed extraction failed: %s", e)
        return default_result


def _finalize_target_description(original_text: str, raw_td: str) -> str:
    cleaned = clean_target_description_for_relevance((raw_td or "").strip())
    return enrich_target_description_from_original(original_text, cleaned)


def understand_task(
    task_description: str,
    model: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    expansion_model: Optional[str] = None,
    expansion_fallback_models: Optional[List[str]] = None,
    use_detailed_extraction: bool = True,
    stability_tries: int = 3,
    only_target_description: bool = False,
) -> Dict[str, Any]:
    """
    Main function.

    Behavior:
    - detailed_analysis keeps the LLM extraction (TARGET_DESCRIPTION cleaned for relevance)
    - if only_target_description=True, return cleaned TARGET_DESCRIPTION only
    """
    if not task_description or not task_description.strip():
        return {
            "structured_output": {} if not only_target_description else {"TARGET_DESCRIPTION": ""},
            "keywords_list": "",
            "stability_score": 0.0,
            "raw_text": "",
            "error": "Empty task description"
        }

    task_model = _resolve_primary_model(
        model,
        "OPENROUTER_TASK_UNDERSTANDING_MODEL",
    )
    task_fallbacks = _resolve_fallback_models(
        fallback_models,
        "OPENROUTER_TASK_UNDERSTANDING_FALLBACK_MODELS",
    )
    target_expansion_model = _resolve_primary_model(
        expansion_model,
        "OPENROUTER_TARGET_EXPANSION_MODEL",
    )
    target_expansion_fallbacks = _resolve_fallback_models(
        expansion_fallback_models,
        "OPENROUTER_TARGET_EXPANSION_FALLBACK_MODELS",
    )

    try:
        if use_detailed_extraction:
            detailed_info = extract_detailed_task_info(
                task_description,
                model=task_model,
                fallback_models=task_fallbacks,
            )

            # Keep LLM extraction for debugging/analysis
            detailed_analysis = copy.deepcopy(detailed_info)

            if only_target_description:
                raw_td = _finalize_target_description(task_description, detailed_analysis.get("TARGET_DESCRIPTION", ""))
                return {
                    "structured_output": {"TARGET_DESCRIPTION": raw_td},
                    "keywords_list": "",
                    "stability_score": 1.0,
                    "raw_text": str(detailed_analysis),
                    "detailed_analysis": detailed_analysis,
                }

            structured_output = copy.deepcopy(detailed_info)

            initial_result = {
                "structured_output": structured_output,
                "keywords_list": "",
                "stability_score": 1.0,
                "raw_text": str(detailed_analysis),
                "detailed_analysis": detailed_analysis,
            }

            result = post_process_extraction(initial_result, task_description)

            target_raw = result.get("structured_output", {}).get("TARGET_OBJECT", "")
            if target_raw and target_raw.lower() not in ["none", "not specified", ""]:
                expanded = expand_target_object_with_synonyms(
                    target_raw,
                    model=target_expansion_model,
                    fallback_models=target_expansion_fallbacks,
                )
                if expanded:
                    result["structured_output"]["TARGET_OBJECT"] = expanded
                    logger.debug("TARGET_OBJECT expanded: %s -> %s chars", len(target_raw), len(expanded))

            result["keywords_list"] = convert_to_keywords_list(result["structured_output"])
            return result

        extracted_text, stability_score = extract_task_info_with_stability(
            task_description,
            model=task_model,
            fallback_models=task_fallbacks,
            tries=stability_tries
        )

        parsed_data = parse_structured_output(extracted_text)
        detailed_analysis = copy.deepcopy(parsed_data)

        if only_target_description:
            raw_td = _finalize_target_description(task_description, detailed_analysis.get("TARGET_DESCRIPTION", ""))
            return {
                "structured_output": {"TARGET_DESCRIPTION": raw_td},
                "keywords_list": "",
                "stability_score": stability_score,
                "raw_text": extracted_text,
                "detailed_analysis": detailed_analysis,
            }

        structured_output = copy.deepcopy(parsed_data)

        initial_result = {
            "structured_output": structured_output,
            "keywords_list": "",
            "stability_score": stability_score,
            "raw_text": extracted_text,
            "detailed_analysis": detailed_analysis,
        }

        result = post_process_extraction(initial_result, task_description)

        target_raw = result.get("structured_output", {}).get("TARGET_OBJECT", "")
        if target_raw and target_raw.lower() not in ["none", "not specified", ""]:
            expanded = expand_target_object_with_synonyms(
                target_raw,
                model=target_expansion_model,
                fallback_models=target_expansion_fallbacks,
            )
            if expanded:
                result["structured_output"]["TARGET_OBJECT"] = expanded
                logger.debug("TARGET_OBJECT expanded: %s -> %s chars", len(target_raw), len(expanded))

        result["keywords_list"] = convert_to_keywords_list(result["structured_output"])
        return result

    except Exception as e:
        logger.exception("Task understanding failed: %s", e)
        return {
            "structured_output": {} if not only_target_description else {"TARGET_DESCRIPTION": ""},
            "keywords_list": "",
            "stability_score": 0.0,
            "raw_text": "",
            "error": str(e)
        }


# ===============================
# CONVENIENCE FUNCTIONS
# ===============================

def get_task_restrictions(task_description: str, model: Optional[str] = None) -> str:
    result = understand_task(task_description, model=model)
    restrictions = result.get('structured_output', {}).get('RESTRICTIONS', 'None')
    return restrictions if restrictions else "None"


def get_task_keywords(task_description: str, model: Optional[str] = None) -> str:
    result = understand_task(
        task_description,
        model=model,
        only_target_description=True,
    )
    return ((result.get("structured_output") or {}).get("TARGET_DESCRIPTION", "") or "").strip()

def get_target_object(task_description: str, model: Optional[str] = None) -> str:
    result = understand_task(task_description, model=model)
    target = result.get('structured_output', {}).get('TARGET_OBJECT', '')
    return target if target else ""


def get_clean_target_description(task_description: str, model: Optional[str] = None) -> str:
    """Direct helper if you only care about the cleaned TARGET_DESCRIPTION."""
    result = understand_task(task_description, model=model, only_target_description=True)
    return (result.get("structured_output") or {}).get("TARGET_DESCRIPTION", "") or ""


# ===============================
# CLI (only when run as script)
# ===============================

if __name__ == "__main__":
    import sys
    example_task = "Photograph a beehive with numerous bees actively moving around. Capture high-resolution video. Required: Workers with ≥3 years photography experience, worker rating ≥4.5, high-reliability devices (≥64GB storage), geotagging enabled, and located within 50 miles."
    result = understand_task(example_task, only_target_description=True)
    if result.get("error"):
        print("Error:", result["error"])
        sys.exit(1)
    print("result:", result)
    print("TARGET_DESCRIPTION:", result["structured_output"]["TARGET_DESCRIPTION"])
