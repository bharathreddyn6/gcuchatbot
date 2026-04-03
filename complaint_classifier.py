"""
Complaint Classifier & Router
Keyword-based classification with optional Gemini LLM fallback.
"""

import os
from typing import Tuple

# Category → (keywords, department)
CATEGORY_MAP = {
    "Technical": {
        "keywords": ["ac", "fan", "projector", "electrical", "light", "socket",
                     "switch", "power", "voltage", "wiring", "generator", "heater"],
        "department": "technical_staff",
    },
    "Cleaning": {
        "keywords": ["garbage", "dirty", "washroom", "toilet", "dust", "sweep",
                     "clean", "waste", "trash", "odor", "smell", "hygiene", "mess"],
        "department": "cleaning_head",
    },
    "Maintenance": {
        "keywords": ["broken", "furniture", "desk", "chair", "door", "window",
                     "wall", "crack", "leak", "tap", "pipe", "roof", "floor",
                     "bench", "ceiling", "damaged", "repair"],
        "department": "maintenance_team",
    },
    "IT": {
        "keywords": ["wifi", "internet", "computer", "system", "network", "lab",
                     "printer", "server", "slow", "connection", "software",
                     "hardware", "monitor", "keyboard", "mouse", "router"],
        "department": "IT_support",
    },
}

DEFAULT_CATEGORY = "Other"
DEFAULT_DEPARTMENT = "admin"


def classify_complaint(description: str) -> Tuple[str, str, float]:
    """
    Classify a complaint description using keyword matching.

    Returns:
        (category, department, confidence)
        confidence is 1.0 for keyword match, 0.5 for LLM, 0.0 for default.
    """
    text = description.lower()

    # Count hits per category
    scores = {}
    for cat, info in CATEGORY_MAP.items():
        score = sum(1 for kw in info["keywords"] if kw in text)
        if score > 0:
            scores[cat] = score

    if scores:
        best_cat = max(scores, key=scores.get)
        dept = CATEGORY_MAP[best_cat]["department"]
        confidence = min(1.0, scores[best_cat] / 3)
        return best_cat, dept, round(confidence, 2)

    # Fallback to Gemini if no keyword match
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.prompts import PromptTemplate

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No API key")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=api_key,
        )
        prompt = PromptTemplate(
            input_variables=["desc"],
            template="""Classify this college complaint into one of these categories:
Technical, Cleaning, Maintenance, IT, Other.

Complaint: {desc}

Reply with ONLY the category name (one word).""",
        )
        result = (prompt | llm).invoke({"desc": description})
        cat = result.content.strip().capitalize()

        if cat in CATEGORY_MAP:
            dept = CATEGORY_MAP[cat]["department"]
            return cat, dept, 0.5

    except Exception as e:
        print(f"[WARN] LLM classifier failed: {e}")

    return DEFAULT_CATEGORY, DEFAULT_DEPARTMENT, 0.0
