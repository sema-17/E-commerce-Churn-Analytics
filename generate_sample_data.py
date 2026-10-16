"""
Generate a realistic *synthetic* e-commerce order dataset.

Use this to test the pipeline. For your final portfolio version, replace
data/raw/orders.csv with a real dataset (Online Retail II, Olist, ...) and
rename its columns to match the schema below.

Schema: order_id, customer_id, order_date, category, quantity, unit_price,
        discount_pct, returned, review_score, country, payment_type
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "orders.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N_CUSTOMERS = 2200
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
TOTAL_DAYS = (END - START).days

COUNTRIES = ["United Kingdom", "India", "USA", "Germany", "France", "Spain"]
COUNTRY_P = [0.35, 0.20, 0.18, 0.11, 0.10, 0.06]
CATEGORIES = {  # (mean price, std)
    "Electronics": (120, 60),
    "Home & Kitchen": (45, 20),
    "Fashion": (35, 15),
    "Beauty": (22, 10),
    "Sports": (50, 25),
    "Books": (14, 5),
}
PAYMENTS = ["Credit Card", "UPI", "PayPal", "Debit Card"]
cat_names = list(CATEGORIES)

rows, order_id = [], 100000
for cid in range(1, N_CUSTOMERS + 1):
    signup = int(rng.integers(0, TOTAL_DAYS - 20))
    unhappy = rng.random() < 0.25                      # more returns, low reviews
    discount_lover = rng.random() < 0.35
    rate = rng.gamma(2.0, 1 / 40)                      # orders per day
    lifetime = rng.exponential(220 if unhappy else 520)  # days until they leave
    country = rng.choice(COUNTRIES, p=COUNTRY_P)
    fav = rng.choice(cat_names)
    payment = rng.choice(PAYMENTS)

    t, first = float(signup), True
    while True:
        if not first:
            t += rng.exponential(1 / rate)
            if t > signup + lifetime or t > TOTAL_DAYS:
                break
        first = False
        cat = fav if rng.random() < 0.6 else rng.choice(cat_names)
        mu, sd = CATEGORIES[cat]
        disc_p = [0.3, 0.15, 0.2, 0.15, 0.2] if discount_lover else [0.7, 0.1, 0.1, 0.05, 0.05]
        order_id += 1
        rows.append({
            "order_id": order_id,
            "customer_id": cid,
            "order_date": (START + pd.Timedelta(days=int(t))).date(),
            "category": cat,
            "quantity": int(rng.integers(1, 4)),
            "unit_price": round(max(3.0, rng.normal(mu, sd)), 2),
            "discount_pct": int(rng.choice([0, 5, 10, 20, 30], p=disc_p)),
            "returned": int(rng.random() < (0.18 if unhappy else 0.04)),
            "review_score": int(np.clip(round(rng.normal(2.8 if unhappy else 4.3, 0.9)), 1, 5)),
            "country": country,
            "payment_type": payment,
        })

df = pd.DataFrame(rows)

# --- add realistic mess so the cleaning step has something to do ---
dups = df.sample(frac=0.01, random_state=1)
df = pd.concat([df, dups], ignore_index=True)
df["customer_id"] = df["customer_id"].astype("float")
df.loc[df.sample(frac=0.01, random_state=2).index, "customer_id"] = np.nan   # missing IDs
df.loc[df.sample(frac=0.015, random_state=3).index, "quantity"] *= -1        # cancellations

df = df.sample(frac=1, random_state=4).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows for ~{N_CUSTOMERS:,} customers -> {OUT}")
