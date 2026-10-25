"""A7 improvement: add retrieved grounding context to the model input.

This is the ONLY change compared with the A6 code in food_classifier.py.
Baseline  : state = {"body": text}
Improved  : state = {"body": text, "reference_notes": [top matching knowledge records]}
The questions, labels, validation, thresholds and fallback rules are unchanged.
"""
import json
import re

import food_classifier as base

with open("knowledge_base.json", encoding="utf-8") as f:
    KB = json.load(f)["records"]


def retrieve(text, top_k=2):
    """Keyword retrieval: score each record by how many of its keywords appear in the text."""
    t = text.lower()
    scored = []
    for rec in KB:
        score = sum(1 for kw in rec["keywords"] if re.search(r"\b" + re.escape(kw) + r"s?\b", t))
        if score > 0:
            scored.append((score, rec))
    scored.sort(key=lambda x: -x[0])  # stable: ties keep knowledge-base order
    return [rec for _, rec in scored[:top_k]]


def call_model_with_context(text, simulate_failure=False):
    if simulate_failure:
        raise RuntimeError("Simulated model outage")
    notes = [f"{r['title']}: {r['note']}" for r in retrieve(text)]
    state = {"body": text}
    if notes:
        state["reference_notes"] = notes  # <-- the single improvement
    result = base.router.predict(state, base.QUESTIONS, model="english")
    a = result["answers"]
    conf = min(
        a["food_type"].get("answer_confidence", a["food_type"]["confidence"]),
        a["calorie_level"].get("answer_confidence", a["calorie_level"]["confidence"]),
    )
    return {
        "food_type": a["food_type"]["choice"],
        "calorie_level": a["calorie_level"]["choice"],
        "confidence": round(float(conf), 3),
    }
