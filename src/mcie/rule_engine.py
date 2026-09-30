from __future__ import annotations

import re
from typing import Any, Dict, List

ALLOWED_INTENTS = {
    "SYMPTOM_REPORT", "SYMPTOM_CHECK", "SYMPTOM_LOCATION", "SYMPTOM_DURATION",
    "MEDICATION_INSTRUCTION", "MEDICATION_QUERY", "ALLERGY_CHECK",
    "MEDICAL_HISTORY", "MEDICAL_PROCEDURE", "FOLLOW_UP_INSTRUCTION",
    "REST_INSTRUCTION", "EMERGENCY", "CONFIRMATION", "HELP_REQUEST",
    "GENERAL_QUESTION", "GENERAL_COMMUNICATION", "UNKNOWN",
}

CONCEPT_ALIASES: Dict[str, List[str]] = {
    "BREATHING_DIFFICULTY": ["susah nafas", "susah bernafas", "susah nak bernafas", "sesak nafas", "difficulty breathing", "shortness of breath"],
    "CHEST_PAIN": ["sakit dada", "chest pain"],
    "STOMACH_PAIN": ["sakit perut", "stomach pain", "abdominal pain"],
    "BACK_PAIN": ["sakit pinggang", "back pain", "waist pain"],
    "SORE_THROAT": ["sakit tekak", "sore throat"],
    "HEADACHE": ["sakit kepala", "headache"],
    "FEVER": ["demam", "fever", "badan panas"],
    "COUGH": ["batuk", "cough"],
    "FLU": ["selesema", "flu", "influenza"],
    "VOMITING": ["muntah", "vomit", "vomiting"],
    "ITCHING": ["gatal-gatal", "gatal", "itching", "itch"],
    "SWELLING": ["bengkak", "swelling", "swollen"],
    "WEAKNESS": ["lemah", "weak"],
    "FATIGUE": ["penat", "tired", "fatigue"],
    "ASTHMA": ["asma", "asthma"],
    "PREGNANCY": ["mengandung", "pregnant", "pregnancy"],
    "ALLERGY": ["alahan", "alergi", "allergy", "allergies", "allergic"],
    "BLOOD_PRESSURE": ["tekanan darah", "blood pressure"],
    "FOLLOW_UP": ["follow-up", "follow up", "datang balik", "datang semula"],
    "EMERGENCY": ["kecemasan", "emergency"],
    "PILL": ["pil", "pill", "tablet"],
    "CREAM": ["krim", "cream"],
    "MEDICATION": ["ubat", "medicine", "medication", "drug"],
    "SURGERY": ["bedah", "surgery"],
    "X_RAY": ["sinar-x", "sinar x", "x-ray", "xray"],
    "PAIN": ["kesakitan", "pain", "sakit"],
}

# Specific concepts must win before generic PAIN.
SPECIFIC_FIRST = [
    "BREATHING_DIFFICULTY", "CHEST_PAIN", "STOMACH_PAIN", "BACK_PAIN",
    "SORE_THROAT", "HEADACHE", "FEVER", "COUGH", "FLU", "VOMITING",
    "ITCHING", "SWELLING", "WEAKNESS", "FATIGUE", "ASTHMA", "PREGNANCY",
    "ALLERGY", "BLOOD_PRESSURE", "FOLLOW_UP", "EMERGENCY", "PILL", "CREAM",
    "MEDICATION", "SURGERY", "X_RAY",
]

NUMBER_WORDS = {
    "zero": 0, "kosong": 0,
    "one": 1, "satu": 1,
    "two": 2, "dua": 2,
    "three": 3, "tiga": 3,
    "four": 4, "empat": 4,
    "five": 5, "lima": 5,
    "six": 6, "enam": 6,
    "seven": 7, "tujuh": 7,
    "eight": 8, "lapan": 8,
    "nine": 9, "sembilan": 9,
    "ten": 10, "sepuluh": 10,
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_medical_concepts(text: str) -> List[str]:
    t = _norm(text)
    out: List[str] = []
    for concept in SPECIFIC_FIRST:
        if any(alias in t for alias in CONCEPT_ALIASES.get(concept, [])):
            out.append(concept)
    # Generic PAIN only when no specific pain concept was already found.
    specific_pain = {"CHEST_PAIN", "STOMACH_PAIN", "BACK_PAIN", "SORE_THROAT", "HEADACHE"}
    if not any(c in specific_pain for c in out):
        if any(alias in t for alias in CONCEPT_ALIASES["PAIN"]):
            out.append("PAIN")
    return list(dict.fromkeys(out))


def _is_question(t: str) -> bool:
    return (
        "?" in t
        or any(x in t for x in [
            "tak", "kah", "apa", "mana", "bila", "berapa", "bagaimana",
            "do you", "are you", "have you", "what", "where", "when", "how long"
        ])
    )


def detect_intent(text: str) -> str:
    t = _norm(text)
    concepts = extract_medical_concepts(t)
    question = _is_question(t)

    if "EMERGENCY" in concepts:
        return "EMERGENCY"

    if any(x in t for x in ["berapa lama", "sejak bila", "bila mula", "how long", "since when"]):
        return "SYMPTOM_DURATION"

    if (
        ("mana" in t and any(x in t for x in ["sakit", "bahagian", "dekat", "kat"]))
        or any(x in t for x in ["where does it hurt", "where is the pain", "where do you feel pain"])
    ):
        return "SYMPTOM_LOCATION"

    if "ALLERGY" in concepts and question:
        return "ALLERGY_CHECK"

    med_words = any(x in t for x in ["ubat", "medicine", "medication", "pill", "pil", "tablet"])
    instruction_words = any(x in t for x in ["ambil", "take", "makan ubat", "gunakan", "use this medicine"])
    query_words = any(x in t for x in ["apa", "what", "which", "tengah ambil", "sedang mengambil", "are you taking"])
    if med_words and instruction_words and not question:
        return "MEDICATION_INSTRUCTION"
    if med_words and (question or query_words):
        return "MEDICATION_QUERY"

    if "BLOOD_PRESSURE" in concepts and any(x in t for x in ["check", "ukur", "measure"]):
        return "MEDICAL_PROCEDURE"
    if any(x in t for x in ["x-ray", "xray", "sinar-x", "sinar x"]) and any(x in t for x in ["buat", "do", "ambil", "take"]):
        return "MEDICAL_PROCEDURE"

    if "FOLLOW_UP" in concepts or any(x in t for x in ["datang balik", "datang semula", "come back", "return next"]):
        return "FOLLOW_UP_INSTRUCTION"

    if any(x in t for x in ["rehat", "rest"]):
        return "REST_INSTRUCTION"

    symptom_concepts = {
        "FEVER", "COUGH", "FLU", "VOMITING", "ITCHING", "SWELLING",
        "WEAKNESS", "FATIGUE", "PAIN", "HEADACHE", "STOMACH_PAIN",
        "BACK_PAIN", "SORE_THROAT", "CHEST_PAIN", "BREATHING_DIFFICULTY", "ASTHMA"
    }
    if any(c in symptom_concepts for c in concepts):
        if question:
            return "SYMPTOM_CHECK"
        return "SYMPTOM_REPORT"

    if question:
        return "GENERAL_QUESTION"
    if t:
        return "GENERAL_COMMUNICATION"
    return "UNKNOWN"


def extract_entities(text: str) -> Dict[str, Any]:
    t = _norm(text)
    concepts = extract_medical_concepts(t)
    symptoms = [
        c for c in concepts
        if c in {
            "FEVER", "COUGH", "FLU", "VOMITING", "ITCHING", "SWELLING",
            "WEAKNESS", "FATIGUE", "PAIN", "HEADACHE", "STOMACH_PAIN",
            "BACK_PAIN", "SORE_THROAT", "CHEST_PAIN", "BREATHING_DIFFICULTY", "ASTHMA"
        }
    ]
    medications = [c for c in concepts if c in {"MEDICATION", "PILL", "CREAM"}]

    quantity = None
    digit = re.search(r"\b(10|[0-9])\b", t)
    if digit:
        quantity = int(digit.group(1))
    else:
        for word, value in NUMBER_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", t):
                quantity = value
                break

    timing = None
    if any(x in t for x in ["lepas makan", "selepas makan", "after meal", "after meals", "after eating"]):
        timing = "AFTER_MEALS"
    elif any(x in t for x in ["sebelum makan", "before meal", "before meals", "before eating"]):
        timing = "BEFORE_MEALS"
    elif any(x in t for x in ["dengan makanan", "with meal", "with meals", "with food"]):
        timing = "WITH_MEALS"

    frequency = None
    if any(x in t for x in ["dua kali sehari", "twice a day", "2 times a day"]):
        frequency = "TWICE_DAILY"
    elif any(x in t for x in ["sekali sehari", "once a day", "1 time a day"]):
        frequency = "ONCE_DAILY"

    return {
        "symptoms": symptoms,
        "medications": medications,
        "quantity": quantity,
        "frequency": frequency,
        "timing": timing,
        "duration": {"quantity": None, "unit": None},
    }


def build_rule_anchor(text: str) -> Dict[str, Any]:
    return {
        "intent": detect_intent(text),
        "medical_concepts": extract_medical_concepts(text),
        "entities": extract_entities(text),
    }
