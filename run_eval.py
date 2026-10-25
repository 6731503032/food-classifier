"""A7 evaluation runner: runs the 10 cases on the baseline (A6) and the improved version.

Usage:  python run_eval.py
Output: terminal tables, eval_log.txt, results_baseline.json, results_improved.json
"""
import json
import math
import statistics
import time

import food_classifier as base
import food_classifier_v2 as v2

RUNS = 5  # each case is run 5 times so p95 latency can be reported (10 cases x 5 = 50 timings)

with open("eval-cases.json", encoding="utf-8") as f:
    CASES = json.load(f)["cases"]

LOG = []


def out(s=""):
    print(s)
    LOG.append(s)


def p95(values):
    s = sorted(values)
    return s[math.ceil(0.95 * len(s)) - 1]  # nearest-rank method


def check(case, r):
    b = case["expected_behavior"]
    if b == "reject":
        return "error" in r
    if "error" in r:
        return False
    labels_ok = (r.get("food_type") == case["expected_food_type"]
                 and r.get("calorie_level") == case["expected_calorie_level"])
    if b == "fallback":
        return labels_ok and r.get("source") == "fallback"
    return labels_ok and r.get("source") == "laya"


def describe(r):
    if "error" in r:
        return "REJECTED: " + r["error"]
    s = f"{r['food_type']}, {r['calorie_level']} (conf {r['confidence']}, {r['source']}"
    if r.get("needs_review"):
        s += ", needs_review"
    return s + ")"


def evaluate(label):
    base.classify_food("warm up apple")  # warm-up call, not measured
    rows, all_ms, model_ms = [], [], []
    for c in CASES:
        runs, times = [], []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            r = base.classify_food(c["input"], simulate_failure=c["simulate_failure"])
            times.append((time.perf_counter() - t0) * 1000)
            runs.append(r)
        r = runs[0]
        all_ms.extend(times)
        if c["expected_behavior"] == "model":
            model_ms.extend(times)
        review_ok = bool(r.get("needs_review", False)) == c["expected_review"]
        ctx = [x["id"] for x in v2.retrieve(c["input"].strip())] if (label == "improved" and c["input"].strip() and not c["simulate_failure"]) else []
        rows.append({
            "id": c["case_id"], "case_type": c["case_type"], "input": c["input"],
            "expected": (f"{c['expected_food_type']}, {c['expected_calorie_level']}"
                         if c["expected_behavior"] != "reject" else "REJECTED: Invalid input"),
            "actual": describe(r), "raw_result": r, "pass": check(c, r), "review_ok": review_ok,
            "median_ms": round(statistics.median(times), 1), "all_ms": [round(t, 1) for t in times],
            "same_result_all_runs": all(x == r for x in runs), "context_records": ctx,
        })
    summary = {
        "label": label, "correct": sum(x["pass"] for x in rows), "total": len(rows),
        "accuracy_pct": 100.0 * sum(x["pass"] for x in rows) / len(rows),
        "review_correct": sum(x["review_ok"] for x in rows),
        "p95_ms_all": round(p95(all_ms), 1), "p95_ms_model_cases": round(p95(model_ms), 1),
        "median_ms_all": round(statistics.median(all_ms), 1), "timings": len(all_ms),
    }
    with open(f"results_{label}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2, ensure_ascii=False)
    return rows, summary


def show(rows, s):
    out(f"\n===== {s['label'].upper()} RESULTS =====")
    for x in rows:
        ctx = f"  context={x['context_records']}" if x["context_records"] else ""
        out(f"{x['id']} | {'PASS' if x['pass'] else 'FAIL'} | {x['median_ms']:>7} ms | "
            f"in={x['input']!r}")
        out(f"      expected: {x['expected']}")
        out(f"      actual  : {x['actual']}{ctx}")
    out(f"Accuracy: {s['correct']} / {s['total']} = {s['accuracy_pct']:.0f}%")
    out(f"p95 latency (all {s['timings']} timings): {s['p95_ms_all']} ms")
    out(f"p95 latency (model cases only): {s['p95_ms_model_cases']} ms")
    out(f"Review-flag behaviour as expected: {s['review_correct']} / {s['total']}")


orig_call_model = base.call_model
b_rows, b_sum = evaluate("baseline")
show(b_rows, b_sum)

base.call_model = v2.call_model_with_context  # the single improvement
i_rows, i_sum = evaluate("improved")
show(i_rows, i_sum)
base.call_model = orig_call_model

out("\n===== BEFORE / AFTER =====")
out(f"Accuracy      : {b_sum['accuracy_pct']:.0f}% -> {i_sum['accuracy_pct']:.0f}%  "
    f"({i_sum['accuracy_pct'] - b_sum['accuracy_pct']:+.0f} points)")
out(f"p95 latency   : {b_sum['p95_ms_all']} ms -> {i_sum['p95_ms_all']} ms  "
    f"({i_sum['p95_ms_all'] - b_sum['p95_ms_all']:+.1f} ms)")
out("Cost per task : N/A API fee. Laya runs locally, so $0.00 per request (computer power not measured).")
for b, i in zip(b_rows, i_rows):
    change = "same" if b["pass"] == i["pass"] else ("IMPROVED" if i["pass"] else "WORSE")
    out(f"{b['id']}: {'PASS' if b['pass'] else 'FAIL'} -> {'PASS' if i['pass'] else 'FAIL'}  [{change}]")

with open("eval_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
out("\nSaved: eval_log.txt, results_baseline.json, results_improved.json")
