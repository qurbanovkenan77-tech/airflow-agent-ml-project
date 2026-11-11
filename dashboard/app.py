
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.workflow import graph

st.set_page_config(
    page_title="Agent ML Dashboard",
    layout="wide"
)

st.title("Agent-Based Machine Learning")
st.write("Life expectancy prediction using cleaned Gapminder data")

st.subheader("Agent Workflow")

dot = """
digraph {
    rankdir=TB;
    node [shape=box, style=rounded];

    START -> data_agent;
    data_agent -> training_agent;
    training_agent -> evaluation_agent;
    evaluation_agent -> decision_agent;
    decision_agent -> training_agent [label="Retry"];
    decision_agent -> reporting_agent [label="Target met / Max attempts"];
    reporting_agent -> END;
}
"""

st.graphviz_chart(dot, use_container_width=True)

st.subheader("Run the Agents")

target = st.slider(
    "Target R²",
    min_value=0.50,
    max_value=0.99,
    value=0.85,
    step=0.01
)

if st.button("Run Agent Workflow"):
    with st.spinner("Training and evaluating models..."):
        graph.invoke(
            {"target_r2": target},
            config={"recursion_limit": 20}
        )
    st.success("Workflow completed successfully!")

results_path = ROOT / "output" / "agent_results.json"

if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    st.subheader("Model Evaluation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Final Model", results["final_model"])
    col2.metric("R² Score", results["final_r2"])
    col3.metric("MAE (Years)", results["final_mae"])

    st.subheader("Training History")

    history = pd.DataFrame(results["history"])
    st.dataframe(history, use_container_width=True)

    st.subheader("Model Performance Comparison")
    st.bar_chart(history.set_index("model")["r2"])

    if results["target_met"]:
        st.success("Performance target achieved!")
    else:
        st.warning("Maximum training attempts reached.")

else:
    st.info("Run the workflow to generate evaluation results.")

st.divider()
st.header("Bonus: Workflow Comparison")

comparison_path = ROOT / "output" / "workflow_comparison.csv"

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)

    st.dataframe(comparison, use_container_width=True)

    st.subheader("Prediction Quality: R²")
    st.bar_chart(
        comparison.set_index("workflow")["r2"]
    )

    st.subheader("Prediction Error: MAE")
    st.bar_chart(
        comparison.set_index("workflow")["mae"]
    )

    st.subheader("Training Attempts")
    st.bar_chart(
        comparison.set_index("workflow")["attempts"]
    )

    st.info(
        "Adaptive routing reduced training attempts from 3 "
        "to 2 compared with the simple agent workflow. "
        "The fixed Airflow baseline trained one predefined model."
    )
else:
    st.warning(
        "Run evaluation/compare_systems.py first."
    )
