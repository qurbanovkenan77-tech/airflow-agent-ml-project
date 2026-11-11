
import csv
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"

with open(data_dir / "gapminder.csv", newline="") as f:
    rows = list(csv.DictReader(f))

# First table: population and life expectancy
with open(data_dir / "country_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["country", "year", "pop", "lifeExp", "continent"]
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in writer.fieldnames})

# Second table: economic indicators
with open(data_dir / "economics.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["country", "year", "gdpPercap"]
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in writer.fieldnames})

print(f"Created two datasets with {len(rows)} rows each.")
