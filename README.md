
# Airflow and Agent-Based Machine Learning Pipeline

## Project Overview

This project implements an end-to-end data pipeline using Apache Airflow, PostgreSQL, and LangGraph.

The pipeline processes Gapminder country-level data and trains machine learning models to predict life expectancy.

The project contains three components:

1. A scheduled Airflow ETL and machine learning pipeline.
2. A LangGraph agent workflow with iterative model training.
3. An adaptive multi-agent workflow with dynamic routing, failure recovery, and evaluation.

A Streamlit dashboard displays the agent graph, model performance, and workflow comparisons.

## Technologies

- Python
- Apache Airflow 3.3.1
- PostgreSQL
- Docker and Docker Compose
- LangGraph
- LangSmith
- Scikit-learn
- Pandas
- Streamlit

## Data Sources

Source: Plotly Gapminder dataset

https://github.com/plotly/datasets

The dataset contains 1,704 country-year observations covering 1952–2007.

For the assignment, the original dataset is separated into two related tables:

**Country Statistics**
- Country
- Year
- Continent
- Population
- Life expectancy

**Economic Statistics**
- Country
- Year
- GDP per capita

Both tables originate from the same source and are joined using country and year.

## Part 1: Airflow Data Pipeline

The Airflow DAG is located at:

`dags/gapminder_pipeline.py`

The pipeline contains seven tasks:

1. `ingest_data`: Validate source files.
2. `clean_country_data`: Clean demographic information.
3. `clean_economics`: Clean economic information.
4. `merge_datasets`: Join both tables.
5. `load_postgres`: Store the cleaned data in PostgreSQL.
6. `analyze_data`: Calculate average life expectancy by continent.
7. `train_fixed_model`: Train a fixed Random Forest baseline.

### Cleaning and Transformation

The pipeline:
- Converts numerical columns to appropriate types.
- Removes invalid records.
- Removes duplicate country-year records.
- Merges country and economic information.
- Loads the result into PostgreSQL.

### Scheduling and Parallelism

The DAG uses a daily schedule with `catchup=False`.

The two cleaning tasks run independently after ingestion. The analysis and model training tasks also run independently after database loading.

### Pipeline Results

- Processed records: 1,704
- Database: PostgreSQL
- Primary table: `gapminder_clean`
- Successful DAG execution: verified in Airflow UI

## Part 2: Agent-Based Machine Learning

The basic workflow is implemented in:

`agents/workflow.py`

It contains five agents:

1. Data Agent: Reads cleaned PostgreSQL data.
2. Training Agent: Trains a regression model.
3. Evaluation Agent: Calculates MAE and R².
4. Decision Agent: Decides whether another model is needed.
5. Reporting Agent: Saves results.

The workflow evaluates Linear Regression, Decision Tree, and Random Forest models.

If performance is below the R² threshold of 0.85, the graph loops back to the Training Agent.

The workflow stops after meeting the target or exhausting the available models.

### Model Results

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 7.9611 | 0.4486 |
| Decision Tree | 4.3448 | 0.7896 |
| Random Forest | 3.4626 | 0.8591 |

Random Forest achieved the best performance in the completed run.

## Optional Part 3: Adaptive Multi-Agent System

The advanced workflow is implemented in:

`agents/adaptive_workflow.py`

It contains agents responsible for:

- Data loading
- Data validation
- Model selection
- Training
- Evaluation
- Conditional routing
- Reporting

Unlike the basic workflow, model selection uses intermediate model performance to determine the next action.

### Adaptive Routing

The workflow begins with Ridge Regression.

When the model performs below the routing threshold, the Selection Agent chooses a nonlinear alternative.

If model training fails, the workflow records the error and attempts a fallback model.

Execution is limited to three training attempts.

### Adaptive Model Results

| Model | MAE | R² |
|---|---:|---:|
| Ridge Regression | 4.9464 | 0.7472 |
| Random Forest | 3.4626 | 0.8591 |

The adaptive workflow reached the target in two attempts.

### Failure Recovery

An intentional training error was injected into the first attempt.

The workflow:
1. Detected and recorded the failure.
2. Routed execution to a fallback model.
3. Continued model evaluation.
4. Completed with Random Forest.

This demonstrates conditional recovery without restarting the entire workflow.

## Evaluation Suite

Files:

- `evaluation/test_cases.json`
- `evaluation/evaluate_workflows.py`
- `evaluation/evaluation_results.json`
- `evaluation/compare_systems.py`

The initial evaluation suite tests:

1. Target performance reached.
2. Performance below target.
3. Maximum attempts reached.
4. Training failure routing.

Each test defines inputs, expected routing, exact-match scoring, and a pass threshold of 1.0.

All four routing tests passed.

Failure-recovery execution was also verified separately using an injected error.

These tests validate routing logic and demonstrated recovery behavior. They do not constitute an exhaustive evaluation of all possible failures.

## Comparison of Three Workflows

| Workflow | Attempts | R² | MAE |
|---|---:|---:|---:|
| Fixed Airflow ML | 1 | 0.8591 | 3.4626 |
| Simple LangGraph | 3 | 0.8591 | 3.4626 |
| Adaptive LangGraph | 2 | 0.8591 | 3.4626 |

The adaptive workflow used one fewer model-training attempt than the simple LangGraph workflow while achieving equivalent reported predictive performance.

The fixed Airflow baseline trained one predefined Random Forest model. It therefore required only one attempt but did not perform adaptive model selection.

The comparison illustrates orchestration differences rather than proving that the adaptive system is universally faster or more accurate.

## LangSmith Monitoring

LangSmith tracing was enabled to observe agent execution.

The recorded traces provide visibility into agent execution order, intermediate decisions, and retraining loops.

Screenshots of the traces are included in the `screenshots` directory.

## Dashboard

The Streamlit dashboard is located at:

`dashboard/app.py`

It displays:
- Agent workflow graph
- Model evaluation metrics
- Training history
- R² and MAE comparisons
- Training attempts for the three workflows

## Running the Project

### Prerequisites

Install:
- Docker Desktop
- Git
- Python 3.11 or newer

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Configure the Environment

```bash
cp .env.example .env
```

The supplied development configuration uses local-only database credentials. Replace them for any nonlocal deployment.

Set `LANGSMITH_API_KEY` in `.env` only if tracing is needed. Do not commit real API keys.

### 3. Prepare the Datasets

The repository includes the source and split CSV files.

To regenerate the split tables:

```bash
python3 scripts/prepare_data.py
```

### 4. Initialize and Start Airflow

```bash
docker compose build
docker compose up airflow-init
docker compose up -d
```

Open Airflow:

http://localhost:8080

For the default local development configuration, the Airflow credentials are `airflow` / `airflow`.

### 5. Create the Application Database

For a fresh Docker environment, create the database before triggering the pipeline:

```bash
docker compose exec postgres psql -U airflow -d airflow -c "CREATE DATABASE gapminder_ml;"
```

This command is required only if the database does not already exist.

### 6. Run the Airflow Pipeline

Open the Airflow UI.

Find `gapminder_pipeline`, enable it, and trigger a run.

Verify that all seven tasks complete successfully.

### 7. Open Streamlit

```bash
docker compose up -d agent-dashboard
```

Open:

http://localhost:8501

### 8. Run the Simple Agent Workflow

```bash
docker compose exec agent-dashboard python agents/workflow.py
```

### 9. Run the Adaptive Agent Workflow

```bash
docker compose exec agent-dashboard python agents/adaptive_workflow.py
```

### 10. Run Routing Tests

```bash
docker compose exec agent-dashboard python evaluation/evaluate_workflows.py
```

### 11. Reproduce the Comparison

Run the two agent workflows and Airflow DAG first.

Then run:

```bash
docker compose exec agent-dashboard python evaluation/compare_systems.py
```

This requires the baseline JSON files in the `evaluation` folder and the Airflow model results in `output`.

## Screenshots

The `screenshots` directory includes evidence of:

- Successful Airflow execution
- Airflow DAG graph
- LangGraph agent graph
- Model evaluation metrics
- LangSmith tracing
- Three-workflow comparison

## Limitations

The reported machine learning results use a random row-level train/test split.

Because multiple observations belong to the same country, observations from a country can appear in both training and testing.

A country-grouped or chronological split would provide a more rigorous assessment of generalization.

The R² target and routing thresholds are project-defined heuristics.

The adaptive system uses rule-based decisions, not an LLM-driven planning agent.

The workflows were compared primarily by prediction performance and training attempts. Training-attempt count alone is not a reliable measure of total computational cost.

The two input CSV tables were derived from one original dataset rather than obtained from two independent sources.

## Conclusion

This project demonstrates the integration of scheduled data engineering pipelines with iterative and adaptive machine learning workflows.

Airflow provides reliable ETL orchestration, PostgreSQL stores processed data, LangGraph manages conditional model execution, and LangSmith supports execution tracing.

The adaptive system demonstrated dynamic routing, reduced model-training attempts relative to the simple agent workflow, and successful recovery from an injected training failure.
