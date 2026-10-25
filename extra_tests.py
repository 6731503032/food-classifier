from food_classifier import classify_food

print("--- EXTRA TESTS ---")
for t in ["", "asdfgh??"]:
    print(repr(t), "->", classify_food(t))