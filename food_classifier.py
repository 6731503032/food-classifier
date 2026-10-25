from laya import Router

router = Router()  # downloads the model on first use

FOOD_TYPES = ["Fruit/Vegetable", "Protein", "Grains", "Dairy", "Snacks/Sweets", "Beverage"]
CAL_LEVELS = ["Low", "Medium", "High"]

QUESTIONS = {
    "food_type": {
        "type": "choice",
        "instructions": "What type of food is this?",
        "criteria": {
            "Fruit/Vegetable": "fruits, vegetables, salads",
            "Protein": "meat, fish, eggs, tofu, beans",
            "Grains": "rice, bread, pasta, noodles, cereal",
            "Dairy": "milk, cheese, yogurt",
            "Snacks/Sweets": "cakes, candy, chips, desserts",
            "Beverage": "drinks, juice, smoothies, soda",
        },
    },
    "calorie_level": {
        "type": "choice",
        "instructions": "How many calories does a typical serving have?",
        "criteria": {
            "Low": "under 150 calories",
            "Medium": "150 to 400 calories",
            "High": "over 400 calories",
        },
    },
}

def call_model(text, simulate_failure=False):
    if simulate_failure:
        raise RuntimeError("Simulated model outage")  # used only for the fallback demo
    result = router.predict({"body": text}, QUESTIONS, model="english")
    a = result["answers"]
    conf = min(a["food_type"].get("answer_confidence", a["food_type"]["confidence"]),
               a["calorie_level"].get("answer_confidence", a["calorie_level"]["confidence"]))
    return {
        "food_type": a["food_type"]["choice"],
        "calorie_level": a["calorie_level"]["choice"],
        "confidence": round(float(conf), 3),
    }

def validate_output(d):
    return (d.get("food_type") in FOOD_TYPES
            and d.get("calorie_level") in CAL_LEVELS
            and 0 <= d.get("confidence", -1) <= 1)

def fallback_rules(text):
    t = text.lower()
    if any(w in t for w in ["cake", "chocolate", "candy", "cookie", "ice cream"]):
        return {"food_type": "Snacks/Sweets", "calorie_level": "High", "confidence": 0.0, "source": "fallback"}
    if any(w in t for w in ["fried", "burger", "pizza"]):
        return {"food_type": "Protein", "calorie_level": "High", "confidence": 0.0, "source": "fallback"}
    if any(w in t for w in ["apple", "banana", "salad", "vegetable"]):
        return {"food_type": "Fruit/Vegetable", "calorie_level": "Low", "confidence": 0.0, "source": "fallback"}
    return {"food_type": "Unknown", "calorie_level": "Unknown", "confidence": 0.0,
            "source": "fallback", "needs_review": True}

def classify_food(text, simulate_failure=False):
    # Input validation
    if not isinstance(text, str) or not text.strip() or len(text) > 200:
        return {"error": "Invalid input", "source": "validation"}
    # Try the model, retry once
    for _ in range(2):
        try:
            data = call_model(text.strip(), simulate_failure)
            if validate_output(data):
                data["source"] = "laya"
                if data["confidence"] < 0.6:
                    data["needs_review"] = True
                return data
        except Exception:
            pass
    return fallback_rules(text)

if __name__ == "__main__":
    tests = ["Fresh apple", "Fried chicken, 3 pieces", "Plain white rice, 1 cup",
             "Chocolate cake slice", "Smoothie"]
    print("--- NORMAL RUN ---")
    for t in tests:
        print(t, "->", classify_food(t))
    print("--- FALLBACK DEMO (simulated outage) ---")
    for t in tests:
        print(t, "->", classify_food(t, simulate_failure=True))