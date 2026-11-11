
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.adaptive_workflow import routing_agent

TEST_FILE = ROOT / "evaluation" / "test_cases.json"
OUTPUT_FILE = ROOT / "evaluation" / "evaluation_results.json"


def evaluate():
    with open(TEST_FILE) as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        state = case["input"].copy()

        decision = routing_agent(state)
        actual = decision["route"]
        expected = case["expected_route"]

        score = 1.0 if actual == expected else 0.0
        passed = score >= case["pass_threshold"]

        results.append({
            "test_id": case["id"],
            "name": case["name"],
            "expected": expected,
            "actual": actual,
            "score": score,
            "passed": passed
        })

        status = "PASS" if passed else "FAIL"
        print(f"{case['id']}: {case['name']} - {status}")

    passed_count = sum(r["passed"] for r in results)

    summary = {
        "total_tests": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results),
        "results": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nPassed {passed_count}/{len(results)} tests")
    print(f"Results saved to {OUTPUT_FILE}")

    return summary


if __name__ == "__main__":
    summary = evaluate()
    if summary["failed"]:
        sys.exit(1)
