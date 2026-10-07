from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.generate_data import generate_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
REPORT_PATH = BASE_DIR / "models" / "evaluation.txt"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "paperless_billing",
    "senior_citizen",
]


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    return df


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    return pipeline, metrics, classification_report(y_test, preds, zero_division=0)


def main():
    df = load_data()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42)),
    ]

    results = []
    trained = {}
    reports = {}

    for name, candidate in candidates:
        pipeline, metrics, report = evaluate(name, candidate, X_train, X_test, y_train, y_test)
        trained[name] = pipeline
        reports[name] = report
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    report_text = "MODEL COMPARISON\n" + results_df.to_string(index=False)
    report_text += f"\n\nBEST MODEL: {best_name}\n\n"
    report_text += reports[best_name]
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved evaluation to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
