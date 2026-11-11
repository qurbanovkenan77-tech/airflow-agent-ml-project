
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="gapminder_ml",
    user="airflow",
    password="airflow"
)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM gapminder_clean")
count = cursor.fetchone()[0]

print("PostgreSQL connection successful!")
print("Total cleaned records:", count)

cursor.close()
conn.close()
