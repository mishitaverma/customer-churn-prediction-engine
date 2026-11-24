import json
import os
import sys
import joblib
import pandas as pd

# Allow importing modules from src/ cleanly
sys.path.insert(0, os.path.abspath("src"))
from load_telco_to_mysql import clean_telco_data


def test_clean_telco_data_integrity():
    """Verify IBM Telco cleaning preserves all 7,043 rows and fixes blank TotalCharges."""
    df = clean_telco_data("data/telco_churn.csv")

    assert len(df) == 7043, f"Expected 7,043 rows, got {len(df)}"
    assert df["customer_id"].is_unique, "Duplicate customer_id values found"
    assert df["total_charges"].isna().sum() == 0, "Null values remain in total_charges"
    assert set(df["churn"].unique()) == {0, 1}, "Churn target must be binary 0/1"


def test_no_id_or_target_leakage_in_model():
    """Verify neither customer_id nor churn exists in the trained pipeline inputs."""
    model = joblib.load("artifacts/best_model.pkl")
    input_features = set(model.feature_names_in_)

    assert "customer_id" not in input_features, "customer_id leaked into model features"
    assert "churn" not in input_features, "Target column leaked into model features"


def test_prediction_probabilities_and_sanity():
    """Verify predictions are valid probabilities and rank high-risk above low-risk profiles."""
    model = joblib.load("artifacts/best_model.pkl")

    sample_accounts = pd.DataFrame([
        {
            # High-risk profile: Month-to-month, 2 months tenure, Fiber optic, Electronic check
            "gender": "Female", "senior_citizen": 1, "partner": "No", "dependents": "No",
            "tenure": 2, "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes",
            "streaming_movies": "Yes", "contract": "Month-to-month",
            "paperless_billing": "Yes", "payment_method": "Electronic check",
            "monthly_charges": 95.0, "total_charges": 190.0,
        },
        {
            # Low-risk profile: Two-year contract, 60 months tenure, DSL, Auto-Pay, Tech Support
            "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes",
            "tenure": 60, "phone_service": "Yes", "multiple_lines": "Yes",
            "internet_service": "DSL", "online_security": "Yes", "online_backup": "Yes",
            "device_protection": "Yes", "tech_support": "Yes", "streaming_tv": "No",
            "streaming_movies": "No", "contract": "Two year",
            "paperless_billing": "No", "payment_method": "Credit card (automatic)",
            "monthly_charges": 65.0, "total_charges": 3900.0,
        },
    ])

    probs = model.predict_proba(sample_accounts)[:, 1]

    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities must fall between 0 and 1"
    assert probs[0] > probs[1], f"Expected high-risk ({probs[0]:.3f}) > low-risk ({probs[1]:.3f})"


def test_saved_evaluation_metrics_are_valid():
    """Verify serialized holdout evaluation artifacts beat random/majority baselines."""
    with open("artifacts/test_metrics.json", "r") as f:
        metrics = json.load(f)

    assert 0.0 < metrics["tuned_threshold"] < 1.0
    assert metrics["roc_auc"] > 0.75, "ROC-AUC below expected benchmark floor"
    assert metrics["pr_auc"] > 0.50, "PR-AUC below expected benchmark floor"