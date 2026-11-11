
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from airflow.sdk import dag, task

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# File locations inside Docker
DATA = Path("/opt/airflow/data")
OUTPUT = Path("/opt/airflow/output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Local PostgreSQL database
DB = {
    "host": "postgres",
    "dbname": "gapminder_ml",
    "user": "airflow",
    "password": "airflow",
}


@dag(
    dag_id="gapminder_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["assignment", "gapminder", "ml"],
)
def gapminder_pipeline():

    # Task 1: Ingest datasets
    @task
    def ingest_data():
        for name in ["country_stats.csv", "economics.csv"]:
            path = DATA / name

            if not path.is_file():
                raise FileNotFoundError(path)

            print(f"Found dataset: {name}")

        return True

    # Task 2: Clean country statistics
    @task
    def clean_country_data(ready):
        records = []

        with open(DATA / "country_stats.csv") as f:
            for row in csv.DictReader(f):
                try:
                    country = row["country"].strip()
                    year = int(row["year"])
                    population = int(float(row["pop"]))
                    life = float(row["lifeExp"])

                    if country and population > 0 and life > 0:
                        records.append({
                            "country": country,
                            "year": year,
                            "continent": row["continent"].strip(),
                            "population": population,
                            "life_expectancy": life,
                        })

                except (ValueError, TypeError, KeyError):
                    continue

        # Remove duplicate country-year records
        records = list({
            (r["country"], r["year"]): r
            for r in records
        }.values())

        path = OUTPUT / "country_clean.csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "country",
                    "year",
                    "continent",
                    "population",
                    "life_expectancy",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

        print(f"Cleaned country data: {len(records)} rows")

        return str(path)

    # Task 3: Clean economic statistics
    @task
    def clean_economics(ready):
        records = []

        with open(DATA / "economics.csv") as f:
            for row in csv.DictReader(f):
                try:
                    country = row["country"].strip()
                    year = int(row["year"])
                    gdp = float(row["gdpPercap"])

                    if country and gdp > 0:
                        records.append({
                            "country": country,
                            "year": year,
                            "gdp_per_capita": gdp,
                        })

                except (ValueError, TypeError, KeyError):
                    continue

        # Remove duplicate country-year records
        records = list({
            (r["country"], r["year"]): r
            for r in records
        }.values())

        path = OUTPUT / "economics_clean.csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "country",
                    "year",
                    "gdp_per_capita",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

        print(f"Cleaned economics data: {len(records)} rows")

        return str(path)

    # Task 4: Merge both datasets
    @task
    def merge_datasets(country_path, economics_path):

        with open(economics_path) as f:
            economics = {
                (r["country"], r["year"]): r
                for r in csv.DictReader(f)
            }

        merged = []

        with open(country_path) as f:
            for row in csv.DictReader(f):
                key = (row["country"], row["year"])

                if key in economics:
                    row["gdp_per_capita"] = (
                        economics[key]["gdp_per_capita"]
                    )
                    merged.append(row)

        path = OUTPUT / "merged.csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "country",
                    "year",
                    "continent",
                    "population",
                    "life_expectancy",
                    "gdp_per_capita",
                ],
            )

            writer.writeheader()
            writer.writerows(merged)

        print(f"Merged data: {len(merged)} rows")

        return str(path)

    # Task 5: Load merged dataset into PostgreSQL
    @task
    def load_postgres(merged_path):

        with open(merged_path) as f:
            rows = [
                (
                    r["country"],
                    int(r["year"]),
                    r["continent"],
                    int(r["population"]),
                    float(r["life_expectancy"]),
                    float(r["gdp_per_capita"]),
                )
                for r in csv.DictReader(f)
            ]

        with psycopg2.connect(**DB) as conn:
            with conn.cursor() as cur:

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS gapminder_clean (
                        country TEXT,
                        year INTEGER,
                        continent TEXT,
                        population BIGINT,
                        life_expectancy DOUBLE PRECISION,
                        gdp_per_capita DOUBLE PRECISION,
                        PRIMARY KEY (country, year)
                    )
                """)

                execute_values(
                    cur,
                    """
                    INSERT INTO gapminder_clean VALUES %s
                    ON CONFLICT (country, year)
                    DO UPDATE SET
                        continent = EXCLUDED.continent,
                        population = EXCLUDED.population,
                        life_expectancy = EXCLUDED.life_expectancy,
                        gdp_per_capita = EXCLUDED.gdp_per_capita
                    """,
                    rows,
                )

        print(f"Loaded {len(rows)} records into PostgreSQL")

        return len(rows)

    # Task 6: Analyze cleaned database records
    @task
    def analyze_data(row_count):

        with psycopg2.connect(**DB) as conn:
            with conn.cursor() as cur:

                cur.execute("""
                    SELECT
                        continent,
                        ROUND(
                            AVG(life_expectancy)::numeric, 2
                        )
                    FROM gapminder_clean
                    GROUP BY continent
                    ORDER BY continent
                """)

                results = cur.fetchall()

        path = OUTPUT / "continent_analysis.csv"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "continent",
                "average_life_expectancy",
            ])

            writer.writerows(results)

        for continent, average in results:
            print(f"{continent}: {average} years")

        print(f"Analyzed {row_count} records")

        return str(path)

    # Task 7: Fixed ML baseline for bonus comparison
    @task
    def train_fixed_model(row_count):

        # Load cleaned data directly from PostgreSQL
        with psycopg2.connect(**DB) as conn:
            with conn.cursor() as cur:

                cur.execute("""
                    SELECT
                        year,
                        continent,
                        population,
                        gdp_per_capita,
                        life_expectancy
                    FROM gapminder_clean
                    ORDER BY country, year
                """)

                columns = [
                    column[0]
                    for column in cur.description
                ]

                df = pd.DataFrame(
                    cur.fetchall(),
                    columns=columns,
                )

        if df.empty:
            raise ValueError("No data found in PostgreSQL")

        # Encode continent as numeric features
        df = pd.get_dummies(
            df,
            columns=["continent"],
            dtype=int,
        )

        # Separate features and target
        X = df.drop(columns=["life_expectancy"])
        y = df["life_expectancy"]

        # Use the same split as the simple agent workflow
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Fixed Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results = {
            "model": "Fixed Random Forest",
            "mae": round(float(mae), 4),
            "r2": round(float(r2), 4),
            "rows": row_count,
        }

        # Save results for the bonus comparison
        result_path = OUTPUT / "airflow_model_results.json"

        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)

        print("Fixed Airflow ML model results:")
        print(json.dumps(results, indent=4))

        return results

    # ------------------------------------
    # Define pipeline dependencies
    # ------------------------------------

    # Ingestion
    ready = ingest_data()

    # Independent parallel cleaning tasks
    country = clean_country_data(ready)
    economics = clean_economics(ready)

    # Merge
    merged = merge_datasets(country, economics)

    # Database loading
    loaded = load_postgres(merged)

    # Run analysis and ML training in parallel
    analyze_data(loaded)
    train_fixed_model(loaded)


# Register the DAG
gapminder_pipeline()
