
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
OUTPUT = ROOT / "output"


def load_json(path):
    with open(path) as f:
        return json.load(f)


airflow = load_json(OUTPUT / "airflow_model_results.json")
simple = load_json(EVAL / "simple_agent_baseline.json")
adaptive = load_json(EVAL / "adaptive_normal.json")
recovery = load_json(EVAL / "adaptive_failure_recovery.json")

comparison = pd.DataFrame([
    {
        "workflow": "Fixed Airflow ML",
        "model": airflow["model"],
        "attempts": 1,
        "r2": airflow["r2"],
        "mae": airflow["mae"],
        "adaptive_routing": False,
        "failure_recovery_tested": False,
    },
    {
        "workflow": "Simple LangGraph",
        "model": simple["final_model"],
        "attempts": len(simple["history"]),
        "r2": simple["final_r2"],
        "mae": simple["final_mae"],
        "adaptive_routing": False,
        "failure_recovery_tested": False,
    },
    {
        "workflow": "Adaptive LangGraph",
        "model": adaptive["best_model"]["model"],
        "attempts": adaptive["attempts"],
        "r2": adaptive["best_model"]["r2"],
        "mae": adaptive["best_model"]["mae"],
        "adaptive_routing": True,
        "failure_recovery_tested": True,
    },
])

comparison.to_csv(
    OUTPUT / "workflow_comparison.csv",
    index=False,
)

print("\nTHREE-WAY WORKFLOW COMPARISON\n")
print(comparison.to_string(index=False))

print("\nFAILURE RECOVERY HISTORY")
for item in recovery["history"]:
    print(item)

