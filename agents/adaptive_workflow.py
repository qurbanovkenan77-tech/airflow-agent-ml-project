
import json
import os
import time
from pathlib import Path
from typing import TypedDict

import pandas as pd
import psycopg2
from langgraph.graph import StateGraph, START, END
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)


class AdaptiveState(TypedDict, total=False):
    data_path: str
    rows: int
    missing_values: int
    model_name: str
    attempt: int
    mae: float
    r2: float
    target_r2: float
    history: list
    route: str
    error: str


def data_agent(state: AdaptiveState):
    with psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "gapminder_ml"),
        user=os.getenv("DB_USER", "airflow"),
        password=os.environ["DB_PASSWORD"]
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT country, year, continent, population,
                       gdp_per_capita, life_expectancy
                FROM gapminder_clean
                ORDER BY country, year
            """)
            columns = [c[0] for c in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)

    path = OUTPUT / "adaptive_data.csv"
    df.to_csv(path, index=False)

    return {
        "data_path": str(path),
        "rows": len(df),
        "attempt": 0,
        "history": [],
        "error": ""
    }


def validation_agent(state: AdaptiveState):
    df = pd.read_csv(state["data_path"])
    missing = int(df.isna().sum().sum())

    if df.empty:
        raise ValueError("Dataset is empty")

    df = df.dropna()

    if len(df) < 100:
        raise ValueError("Not enough records for modeling")

    df.to_csv(state["data_path"], index=False)

    print(f"Validation: {len(df)} rows, {missing} missing values")
    return {
        "rows": len(df),
        "missing_values": missing
    }



def selection_agent(state: AdaptiveState):
    history = state.get("history", [])
    attempted = {item["model"] for item in history}

    if not history:
        model_name = "Ridge Regression"
        reason = "Start with a regularized linear baseline"

    else:
        successful = [
            item for item in history
            if item["status"] == "success"
        ]

        if not successful:
            candidates = ["Decision Tree", "Random Forest"]
            reason = "Previous training failed; use a fallback model"

        else:
            best = max(successful, key=lambda x: x["r2"])

            if best["r2"] < 0.80:
                candidates = ["Random Forest", "Decision Tree"]
                reason = "Low R²; try a more flexible nonlinear model"
            else:
                candidates = ["Decision Tree", "Random Forest"]
                reason = "Moderate R²; try a simpler nonlinear model"

        model_name = next(
            (name for name in candidates if name not in attempted),
            "Random Forest"
        )

    print(f"Selection Agent: {model_name}")
    print(f"Reason: {reason}")

    return {"model_name": model_name}



def training_agent(state: AdaptiveState):
    try:
        if (
            os.getenv("SIMULATE_TRAINING_FAILURE") == "1"
            and state["attempt"] == 0
        ):
            raise RuntimeError("Simulated training failure")
        df = pd.read_csv(state["data_path"])
        df = pd.get_dummies(df, columns=["continent"], dtype=int)

        # Country identifies records but is not used as a predictor.
        X = df.drop(columns=["country", "life_expectancy"])
        y = df["life_expectancy"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if state["model_name"] == "Ridge Regression":
            model = Ridge(alpha=10.0)
        elif state["model_name"] == "Decision Tree":
            model = DecisionTreeRegressor(
                max_depth=5, random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=8,
                random_state=42, n_jobs=-1
            )

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        pd.DataFrame({
            "actual": y_test.to_numpy(),
            "predicted": predicted
        }).to_csv(OUTPUT / "adaptive_predictions.csv", index=False)

        return {"error": ""}

    except Exception as exc:
        print(f"Training failed: {exc}")
        return {"error": str(exc)}


def training_route(state: AdaptiveState):
    if state.get("error"):
        return "failed"
    return "success"


def evaluation_agent(state: AdaptiveState):
    df = pd.read_csv(OUTPUT / "adaptive_predictions.csv")

    mae = float(mean_absolute_error(df["actual"], df["predicted"]))
    r2 = float(r2_score(df["actual"], df["predicted"]))

    history = state["history"] + [{
        "model": state["model_name"],
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "status": "success"
    }]

    print(f"Evaluation: MAE={mae:.4f}, R²={r2:.4f}")

    return {"mae": mae, "r2": r2, "history": history}


def routing_agent(state: AdaptiveState):
    history = list(state["history"])

    if state.get("error"):
        history.append({
            "model": state["model_name"],
            "status": "failed",
            "error": state["error"]
        })

    if not state.get("error") and state["r2"] >= state["target_r2"]:
        route = "finish"
    elif state["attempt"] >= 2:
        route = "finish"
    else:
        route = "retry"

    print(f"Routing Agent: {route}")

    return {
        "route": route,
        "attempt": state["attempt"] + 1 if route == "retry"
                   else state["attempt"],
        "history": history
    }


def next_route(state: AdaptiveState):
    return state["route"]


def reporting_agent(state: AdaptiveState):
    successful = [
        item for item in state["history"]
        if item["status"] == "success"
    ]

    best = max(successful, key=lambda item: item["r2"]) if successful else None

    report = {
        "best_model": best,
        "attempts": len(state["history"]),
        "rows": state["rows"],
        "missing_values": state["missing_values"],
        "target_r2": state["target_r2"],
        "history": state["history"]
    }

    with open(OUTPUT / "adaptive_results.json", "w") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=2))
    return {}


builder = StateGraph(AdaptiveState)

builder.add_node("data_agent", data_agent)
builder.add_node("validation_agent", validation_agent)
builder.add_node("selection_agent", selection_agent)
builder.add_node("training_agent", training_agent)
builder.add_node("evaluation_agent", evaluation_agent)
builder.add_node("routing_agent", routing_agent)
builder.add_node("reporting_agent", reporting_agent)

builder.add_edge(START, "data_agent")
builder.add_edge("data_agent", "validation_agent")
builder.add_edge("validation_agent", "selection_agent")
builder.add_edge("selection_agent", "training_agent")

builder.add_conditional_edges(
    "training_agent",
    training_route,
    {
        "success": "evaluation_agent",
        "failed": "routing_agent"
    }
)

builder.add_edge("evaluation_agent", "routing_agent")

builder.add_conditional_edges(
    "routing_agent",
    next_route,
    {
        "retry": "selection_agent",
        "finish": "reporting_agent"
    }
)

builder.add_edge("reporting_agent", END)

graph = builder.compile()


if __name__ == "__main__":
    start = time.perf_counter()

    graph.invoke(
        {"target_r2": 0.85},
        config={"recursion_limit": 30}
    )

    elapsed = time.perf_counter() - start

    (OUTPUT / "adaptive_graph.mmd").write_text(
        graph.get_graph().draw_mermaid()
    )

    print(f"\nAdaptive workflow finished in {elapsed:.2f} seconds")
