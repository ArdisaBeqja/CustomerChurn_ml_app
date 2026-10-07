from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "customer_churn.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 85, n_rows)
    tenure_months = rng.integers(0, 73, n_rows)
    monthly_charges = np.round(rng.uniform(20, 120, n_rows), 2)
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.85, 1.05, n_rows), 2)
    support_calls = np.clip(rng.poisson(2.0, n_rows), 0, 10)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.55, 0.25, 0.20],
    )
    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_rows,
    )
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "None"],
        size=n_rows,
        p=[0.5, 0.4, 0.1],
    )
    paperless_billing = rng.choice([True, False], n_rows, p=[0.65, 0.35])
    senior_citizen = age >= 65

    risk = (
        -2.0
        + 0.018 * (monthly_charges - 60)
        - 0.035 * tenure_months
        + 0.32 * support_calls
        + 1.05 * (contract_type == "Month-to-month")
        - 0.65 * (contract_type == "Two year")
        + 0.55 * (payment_method == "Electronic check")
        + 0.45 * (internet_service == "Fiber optic")
        + 0.25 * paperless_billing.astype(int)
        + 0.22 * senior_citizen.astype(int)
    )

    churn_probability = sigmoid(risk)
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["churn"].value_counts(normalize=True).rename("ratio"))
