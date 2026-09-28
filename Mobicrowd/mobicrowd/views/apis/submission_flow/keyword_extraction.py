# keyword extraction utilities
import re
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, util

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def clean_and_deduplicate(text):
    """Remove duplicates and clean text"""
    if not text or text == "Not specified":
        return []
    keywords = [kw.strip().lower() for kw in text.split(",") if kw.strip()]
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    return unique_keywords

def convert_to_keywords_list(parsed_data):
    out = []
    for field, value in parsed_data.items():
        if value == "Not specified":
            continue
        keywords = clean_and_deduplicate(value)
        out.append(f"{field.lower()}: {', '.join(keywords)}")
    return str(out).replace("'", "")

def parse_structured_output(text):
    """Parse LLM output into structured data"""
    fields = [
        "TASK_TOPIC", "ACTION_REQUIRED", "TARGET_OBJECT", "TARGET_DESCRIPTION", "LOCATION_DETAILS",
        "TIME_FRAME", "WORKER_QUALIFICATIONS", "EQUIPMENT_NEEDED", "COMPENSATION",
        "VERIFICATION_METHOD", "SAFETY_ISSUES", "PRIVACY_CONCERNS", "RESTRICTIONS",
        "ADDITIONAL_CONDITIONS", "RELEVANT_KEYWORDS"
    ]
    parsed = OrderedDict((field, "Not specified") for field in fields)
    pattern = r"([A-Z_]+):\s*((?:(?!\n[A-Z_]+:).)*)"
    for match in re.finditer(pattern, text, re.DOTALL):
        key, val = match.group(1).strip(), match.group(2).strip()
        if key in parsed:
            parsed[key] = val
    return parsed

def stability_check(outputs):
    """Check stability of multiple outputs"""
    if len(outputs) < 2:
        return 1.0
    
    # Check text similarity
    def get_words(text):
        return set(text.lower().split())
    
    words_sets = [get_words(output) for output in outputs]
    
    similarities = []
    for i in range(len(words_sets)):
        for j in range(i + 1, len(words_sets)):
            intersection = len(words_sets[i] & words_sets[j])
            union = len(words_sets[i] | words_sets[j])
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
    
    return round(sum(similarities) / len(similarities), 4) if similarities else 0.0
