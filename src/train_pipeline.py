import json
import os
import joblib
import numpy as np
import pandas as pd
import mysql.connector
import shap
from dotenv import load_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

load_dotenv()

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
RANDOM_STATE = 42


def fetch_data() -> pd.DataFrame:
    """Load IBM Telco records from MySQL first; fall back to local CSV if DB is offline."""
    if os.getenv("DB_HOST"):
        try:
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                connection_timeout=3,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM telco_customers;")
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            cursor.close()
            conn.close()

            df = pd.DataFrame(rows, columns=cols)
            for col in ["tenure", "senior_citizen", "churn"]:
                df[col] = df[col].astype(int)
            for col in ["monthly_charges", "total_charges"]:
                df[col] = df[col].astype(float)

            print(f"Loaded {len(df):,} rows from MySQL (telco_customers).")
            return df
        except Exception as exc:
            print(f"MySQL unavailable ({exc}); falling back to data/telco_churn.csv.")

    from load_telco_to_mysql import clean_telco_data
    return clean_telco_data("data/telco_churn.csv")


def run_leakage_checks(df: pd.DataFrame, X: pd.DataFrame, train_idx, test_idx) -> None:
    """Verify unique IDs, zero train/test overlap, and no ID/target columns in features."""
    assert df["customer_id"].is_unique, "Duplicate customer_id detected!"
    assert "customer_id" not in X.columns, "customer_id leaked into features!"
    assert "churn" not in X.columns, "Target column leaked into features!"
    train_ids = set(df.loc[train_idx, "customer_id"])
    test_ids = set(df.loc[test_idx, "customer_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Train/Test customer_id overlap!"
    print("Leakage checks passed: unique IDs, zero train/test overlap, fold-scoped preprocessing.")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )


def get_candidate_models(preprocessor: ColumnTransformer) -> dict:
    return {
        "Majority Baseline": ImbPipeline([
            ("prep", preprocessor),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": ImbPipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression (Balanced)": ImbPipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": ImbPipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Random Forest (Balanced)": ImbPipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Random Forest + SMOTE": ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": ImbPipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
            )),
        ]),
    }


def evaluate_models_cv(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"Model": name}
        for metric in scoring:
            mean_val = scores[f"test_{metric}"].mean()
            std_val = scores[f"test_{metric}"].std()
            row[f"{metric}_mean"] = round(float(mean_val), 4)
            row[f"{metric}_std"] = round(float(std_val), 4)
            row[metric.upper()] = f"{mean_val:.3f} ± {std_val:.3f}"
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="pr_auc_mean", ascending=False)


def tune_threshold(best_pipe, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[float, pd.DataFrame]:
    """Tune cutoff strictly on out-of-fold probabilities from the training split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    records = []
    for t in np.round(np.arange(0.20, 0.71, 0.05), 2):
        preds = (oof_probs >= t).astype(int)
        records.append({
            "threshold": float(t),
            "precision": round(precision_score(y_train, preds, zero_division=0), 4),
            "recall": round(recall_score(y_train, preds, zero_division=0), 4),
            "f1": round(f1_score(y_train, preds, zero_division=0), 4),
            "flagged_pct": round(float(preds.mean() * 100), 2),
        })

    thresh_df = pd.DataFrame(records)
    best_threshold = float(thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"])
    return best_threshold, thresh_df


def compute_shap_and_importance(fitted_pipe, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Compute global SHAP values and native model feature importance."""
    prep = fitted_pipe.named_steps["prep"]
    clf = fitted_pipe.named_steps["clf"]

    feature_names = prep.get_feature_names_out()
    X_train_enc = pd.DataFrame(prep.transform(X_train), columns=feature_names).astype(float)
    X_test_enc = pd.DataFrame(prep.transform(X_test), columns=feature_names).astype(float)

    if isinstance(clf, LogisticRegression):
        background = shap.sample(X_train_enc, min(200, len(X_train_enc)), random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(clf, background)
        shap_exp = explainer(X_test_enc)
        native_importance = np.abs(clf.coef_[0])
    else:
        X_eval = X_test_enc.sample(min(500, len(X_test_enc)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(clf)
        shap_exp = explainer(X_eval, check_additivity=False)
        native_importance = clf.feature_importances_

    values = shap_exp.values
    if values.ndim == 3:
        values = values[:, :, 1]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap_direction": values.mean(axis=0),
        "model_importance": native_importance,
    }).sort_values("mean_abs_shap", ascending=False)

    return shap_df


def main():
    df = fetch_data()
    y = df["churn"].astype(int)
    X = df.drop(columns=["customer_id", "churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    run_leakage_checks(df, X, X_train.index, X_test.index)

    preprocessor = build_preprocessor(X_train)
    models = get_candidate_models(preprocessor)

    print("\nRunning Stratified 5-Fold CV across 7 models (takes ~20-40 sec)...")
    cv_results = evaluate_models_cv(models, X_train, y_train)
    display_cols = ["Model", "PR_AUC", "ROC_AUC", "F1", "RECALL", "PRECISION", "ACCURACY"]
    print("\n=== 5-Fold Stratified CV Results (Sorted by PR-AUC) ===")
    print(cv_results[display_cols].to_string(index=False))
    cv_results.to_csv(f"{ARTIFACTS_DIR}/cv_results.csv", index=False)

    best_model_name = cv_results.iloc[0]["Model"]
    best_pipe = models[best_model_name]
    print(f"\nSelected Best Model (highest PR-AUC): {best_model_name}")

    best_threshold, thresh_df = tune_threshold(best_pipe, X_train, y_train)
    thresh_df.to_csv(f"{ARTIFACTS_DIR}/threshold_analysis.csv", index=False)
    print(f"\nOptimal F1 Threshold (Out-of-Fold): {best_threshold:.2f}")
    print(thresh_df.to_string(index=False))

    # Fit on full 80% training split, evaluate once on untouched 20% test split
    best_pipe.fit(X_train, y_train)
    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_preds_default = (test_probs >= 0.50).astype(int)
    test_preds_tuned = (test_probs >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, test_preds_tuned)
    test_metrics = {
        "best_model": best_model_name,
        "tuned_threshold": best_threshold,
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "default_050": {
            "accuracy": round(float(accuracy_score(y_test, test_preds_default)), 4),
            "precision": round(float(precision_score(y_test, test_preds_default)), 4),
            "recall": round(float(recall_score(y_test, test_preds_default)), 4),
            "f1": round(float(f1_score(y_test, test_preds_default)), 4),
        },
        "tuned": {
            "accuracy": round(float(accuracy_score(y_test, test_preds_tuned)), 4),
            "precision": round(float(precision_score(y_test, test_preds_tuned)), 4),
            "recall": round(float(recall_score(y_test, test_preds_tuned)), 4),
            "f1": round(float(f1_score(y_test, test_preds_tuned)), 4),
        },
        "confusion_matrix_tuned": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    print("\n=== Holdout Test Set Evaluation (20% Unseen Data) ===")
    print(json.dumps(test_metrics, indent=2))
    with open(f"{ARTIFACTS_DIR}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nComputing SHAP & feature importance on test set...")
    shap_df = compute_shap_and_importance(best_pipe, X_train, X_test)
    shap_df.to_csv(f"{ARTIFACTS_DIR}/shap_importance.csv", index=False)
    print("\nTop 10 Churn Drivers (Mean |SHAP|):")
    print(shap_df.head(10).to_string(index=False))

    joblib.dump(best_pipe, f"{ARTIFACTS_DIR}/best_model.pkl")
    print(f"\nDone! Saved model and all evaluation files to '{ARTIFACTS_DIR}/'.")


if __name__ == "__main__":
    main()