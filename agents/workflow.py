
import json
import os
from pathlib import Path
from typing import TypedDict

import pandas as pd
import psycopg2
from langgraph.graph import StateGraph, START, END
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

MODELS = ["Linear Regression", "Decision Tree", "Random Forest"]


class AgentState(TypedDict, total=False):
    attempt: int
    model_name: str
    mae: float
    r2: float
    target_r2: float
    data_path: str
    history: list
    completed: bool


# Agent 1: Load cleaned data from PostgreSQL
def data_agent(state: AgentState):
    with psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "gapminder_ml"),
        user=os.getenv("DB_USER", "airflow"),
        password=os.environ["DB_PASSWORD"],
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT year, continent, population,
                       gdp_per_capita, life_expectancy
                FROM gapminder_clean
                ORDER BY country, year
            """)
            columns = [col[0] for col in cursor.description]
            data = pd.DataFrame(cursor.fetchall(), columns=columns)

    if data.empty:
        raise ValueError("Database returned no data")

    data = data.dropna()
    data = pd.get_dummies(data, columns=["continent"], dtype=int)

    path = OUTPUT / "ml_data.csv"
    data.to_csv(path, index=False)

    print(f"Data Agent: Loaded {len(data)} records")
    return {"data_path": str(path), "attempt": 0, "history": []}


# Agent 2: Train a model
def training_agent(state: AgentState):
    data = pd.read_csv(state["data_path"])

    X = data.drop(columns=["life_expectancy"])
    y = data["life_expectancy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    attempt = state["attempt"]
    model_name = MODELS[attempt]

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(
            max_depth=5, random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=42, n_jobs=-1
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Save predictions for the Evaluation Agent
    results = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions
    })
    results.to_csv(OUTPUT / "predictions.csv", index=False)

    print(f"Training Agent: Trained {model_name}")
    return {"model_name": model_name}


# Agent 3: Evaluate the trained model
def evaluation_agent(state: AgentState):
    results = pd.read_csv(OUTPUT / "predictions.csv")

    mae = mean_absolute_error(
        results["actual"], results["predicted"]
    )
    r2 = r2_score(
        results["actual"], results["predicted"]
    )

    history = state["history"] + [{
        "model": state["model_name"],
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4)
    }]

    print(f"Evaluation Agent: MAE={mae:.4f}, R²={r2:.4f}")

    return {
        "mae": float(mae),
        "r2": float(r2),
        "history": history
    }


# Agent 4: Decide whether to retry
def decision_agent(state: AgentState):
    passed = state["r2"] >= state["target_r2"]
    last_model = state["attempt"] >= len(MODELS) - 1

    if passed:
        print("Decision Agent: Performance target reached")
    elif last_model:
        print("Decision Agent: Maximum attempts reached")
    else:
        print("Decision Agent: Trying another model")

    return {
        "completed": passed or last_model,
        "attempt": state["attempt"] if passed or last_model
                   else state["attempt"] + 1
    }


def route_after_decision(state: AgentState):
    return "report" if state["completed"] else "retry"


# Agent 5: Save evaluation report
def reporting_agent(state: AgentState):
    report = {
        "final_model": state["model_name"],
        "final_mae": round(state["mae"], 4),
        "final_r2": round(state["r2"], 4),
        "target_r2": state["target_r2"],
        "target_met": state["r2"] >= state["target_r2"],
        "history": state["history"]
    }

    with open(OUTPUT / "agent_results.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\nReporting Agent: Final results")
    print(json.dumps(report, indent=2))
    return {}


# Build the agent graph
builder = StateGraph(AgentState)

builder.add_node("data_agent", data_agent)
builder.add_node("training_agent", training_agent)
builder.add_node("evaluation_agent", evaluation_agent)
builder.add_node("decision_agent", decision_agent)
builder.add_node("reporting_agent", reporting_agent)

builder.add_edge(START, "data_agent")
builder.add_edge("data_agent", "training_agent")
builder.add_edge("training_agent", "evaluation_agent")
builder.add_edge("evaluation_agent", "decision_agent")

builder.add_conditional_edges(
    "decision_agent",
    route_after_decision,
    {
        "retry": "training_agent",
        "report": "reporting_agent"
    }
)

builder.add_edge("reporting_agent", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"target_r2": 0.85},
        config={"recursion_limit": 20}
    )

    mermaid = graph.get_graph().draw_mermaid()
    (OUTPUT / "agent_graph.mmd").write_text(mermaid)

    print("\nWorkflow completed!")
    print("Results saved to output/agent_results.json")
    print("Graph saved to output/agent_graph.mmd")
