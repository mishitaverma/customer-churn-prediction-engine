# Data Sources & Attribution

## 1. Primary Analytics & Modeling Dataset — IBM Telco Customer Churn
* **Source:** IBM Sample Datasets / Kaggle (`blastchar/telco-customer-churn`)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Size:** 7,043 customer records across 21 attributes (demographics, account tenure, contract type, services, billing, and churn label).
* **Important Note:** This is a **public benchmark dataset** representing a fictional California telecommunications company—not real-world customer PII. All metrics, SQL views, and model evaluations in StayMetric are computed strictly on this dataset.
* **Known Data Quirk Handled:** 11 rows with `tenure = 0` contain blank whitespace (`" "`) in `TotalCharges`. During ingestion, these are coerced to `0.0` (consistent with zero completed billing cycles) rather than dropped, preserving all 7,043 records.

## 2. Interactive Risk Simulator — Synthetic Demo Data (`synthetic_demo/`)
* **Location:** `synthetic_demo/generate_synthetic_demo.py` and `synthetic_demo/synthetic_demo_customers.csv`
* **Purpose:** Rule-based synthetic records generated via Python `Faker` exclusively for sandbox/demo stress-testing in the simulator tab.
* **Scope Boundary:** Synthetic demo data is **never** mixed into SQL analytics views, model training, or reported evaluation metrics.