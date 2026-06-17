"""Quick DB check — run on Windows PC to verify what's in Railway PostgreSQL."""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
db_url = os.environ["DATABASE_URL"]
conn = psycopg2.connect(db_url, sslmode="require")

with conn.cursor() as cur:
    cur.execute("SELECT symbol, spread, updated_at FROM spreads ORDER BY symbol")
    rows = cur.fetchall()

print(f"\n{'Symbol':<12} {'Spread':>12} {'Updated At'}")
print("-" * 50)
for sym, spread, updated_at in rows:
    print(f"{sym:<12} {spread:>12.5f}  {updated_at}")

with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM bars")
    n = cur.fetchone()[0]
print(f"\nTotal bars in DB: {n:,}")

conn.close()
