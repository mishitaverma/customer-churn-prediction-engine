import json
import math
import os
import joblib
import mysql.connector
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="StayMetric — Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

FEATURE_COLS = [
    "gender", "senior_citizen", "partner", "dependents",
    "tenure", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges",
]


def _clean_telco_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
    df.columns = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies", "contract", "paperless_billing",
        "payment_method", "monthly_charges", "total_charges", "churn",
    ]
    return df


def _add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 1000],
        labels=["0-12 Months", "13-24 Months", "25-48 Months", "49+ Months"],
    ).astype(str)
    return df


def _compute_views_pandas(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mirror the exact MySQL CTE/window-function views when running in cloud CSV mode."""
    d = _add_tenure_bucket(df)

    # v_churn_by_tenure
    t = (
        d.groupby("tenure_bucket", as_index=False)
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churn", "sum"),
            churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            monthly_revenue_lost=("monthly_charges", lambda s: round(s[d.loc[s.index, "churn"] == 1].sum(), 2)),
        )
    )
    t["pct_of_all_churners"] = (100.0 * t["churned_customers"] / t["churned_customers"].sum()).round(2)
    t["risk_rank"] = t["churn_rate_pct"].rank(ascending=False, method="min").astype(int)
    t = t.sort_values("risk_rank")

    # v_churn_by_payment
    company_avg = 100.0 * d["churn"].sum() / len(d)
    p = (
        d.groupby("payment_method", as_index=False)
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churn", "sum"),
            churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            avg_monthly_charge=("monthly_charges", lambda s: round(s.mean(), 2)),
        )
    )
    p["diff_vs_company_avg_pct"] = (p["churn_rate_pct"] - company_avg).round(2)
    p["risk_rank"] = p["churn_rate_pct"].rank(ascending=False, method="min").astype(int)
    p = p.sort_values("risk_rank")

    # v_churn_by_segment
    s_df = (
        d.groupby(["contract", "tenure_bucket", "internet_service"], as_index=False)
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churn", "sum"),
            churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            monthly_revenue_lost=("monthly_charges", lambda s: round(s[d.loc[s.index, "churn"] == 1].sum(), 2)),
        )
    )
    total_rev_lost = s_df["monthly_revenue_lost"].sum()
    s_df["pct_of_revenue_lost"] = (100.0 * s_df["monthly_revenue_lost"] / total_rev_lost).round(2)
    s_df["risk_rank"] = s_df["churn_rate_pct"].rank(ascending=False, method="dense").astype(int)
    s_df = s_df.sort_values(["risk_rank", "monthly_revenue_lost"], ascending=[True, False])

    return {
        "v_churn_by_tenure": t,
        "v_churn_by_payment": p,
        "v_churn_by_segment": s_df,
    }


def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_numeric(df[col])
                df[col] = converted
            except (ValueError, TypeError):
                pass
    return df


@st.cache_data(show_spinner=False)
def load_data_and_views():
    """Connect to MySQL views locally; fall back cleanly to IBM CSV on Streamlit Cloud."""
    if os.getenv("DB_HOST"):
        try:
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                connection_timeout=2,
            )
            cursor = conn.cursor()

            def read_query(q: str) -> pd.DataFrame:
                cursor.execute(q)
                frame = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
                return _coerce_numeric_cols(frame)

            df = read_query("SELECT * FROM telco_customers;")
            views = {
                "v_churn_by_tenure": read_query("SELECT * FROM v_churn_by_tenure;"),
                "v_churn_by_payment": read_query("SELECT * FROM v_churn_by_payment;"),
                "v_churn_by_segment": read_query("SELECT * FROM v_churn_by_segment;"),
            }
            cursor.close()
            conn.close()
            return _add_tenure_bucket(df), views, "MySQL Live (telco_customers + SQL Views)"
        except Exception:
            pass

    df = _clean_telco_csv("data/telco_churn.csv")
    views = _compute_views_pandas(df)
    return _add_tenure_bucket(df), views, "IBM Telco CSV Fallback (Cloud Mode)"


@st.cache_resource(show_spinner=False)
def load_model_and_artifacts():
    model = joblib.load("artifacts/best_model.pkl")
    cv_df = pd.read_csv("artifacts/cv_results.csv")
    thresh_df = pd.read_csv("artifacts/threshold_analysis.csv")
    shap_df = pd.read_csv("artifacts/shap_importance.csv")
    with open("artifacts/test_metrics.json", "r") as f:
        test_metrics = json.load(f)
    return model, cv_df, thresh_df, shap_df, test_metrics


@st.cache_data(show_spinner=False)
def score_accounts(_model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scored = df.copy()
    probs = _model.predict_proba(scored[FEATURE_COLS])[:, 1]
    scored["churn_prob"] = probs
    scored["high_risk"] = (scored["churn_prob"] >= threshold).astype(int)
    scored["monthly_rev_at_risk"] = scored["churn_prob"] * scored["monthly_charges"]
    scored["annual_rev_at_risk"] = scored["monthly_rev_at_risk"] * 12
    return scored


def calc_ab_sample_size(p1: float, mde: float = 0.05, alpha: float = 0.05, power: float = 0.80) -> int:
    """Two-proportion z-test sample size per arm (two-sided)."""
    z_alpha = 1.96  # alpha = 0.05
    z_beta = 0.84   # power = 0.80
    p2 = max(0.01, p1 - mde)
    numerator = ((z_alpha + z_beta) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(numerator / (mde ** 2))


# --- Load Everything ---
raw_df, sql_views, data_source_label = load_data_and_views()
model, cv_df, thresh_df, shap_df, test_metrics = load_model_and_artifacts()
tuned_threshold = float(test_metrics["tuned_threshold"])
scored_df = score_accounts(model, raw_df, tuned_threshold)

# --- Header ---
st.title("StayMetric — Customer Churn Intelligence")
st.caption(
    f"Public Benchmark: IBM Telco Customer Churn (7,043 records, CC BY 4.0)  |  "
    f"Source: {data_source_label}  |  Active Model: {test_metrics['best_model']} (Threshold: {tuned_threshold:.2f})"
)

# --- Sidebar Filters ---
st.sidebar.header("Segment Filters")
contracts = sorted(scored_df["contract"].unique().tolist())
tenures = ["0-12 Months", "13-24 Months", "25-48 Months", "49+ Months"]
plans = sorted(scored_df["internet_service"].unique().tolist())

sel_contracts = st.sidebar.multiselect("Contract Type", contracts, default=contracts)
sel_tenures = st.sidebar.multiselect("Tenure Bucket", tenures, default=tenures)
sel_plans = st.sidebar.multiselect("Internet Plan", plans, default=plans)

filtered_df = scored_df[
    scored_df["contract"].isin(sel_contracts)
    & scored_df["tenure_bucket"].isin(sel_tenures)
    & scored_df["internet_service"].isin(sel_plans)
]

st.sidebar.divider()
st.sidebar.markdown(
    "**Data Attribution**\n\n"
    "Primary analytics use the public **IBM Telco Customer Churn** benchmark (fictional company, CC BY 4.0). "
    "See `DATA_SOURCE.md` for full details."
)

# --- Navigation Tabs ---
tab_exec, tab_model, tab_sql, tab_sim = st.tabs([
    "1. Executive & Revenue at Risk",
    "2. Model Evaluation & SHAP",
    "3. SQL Insights (CTEs & Windows)",
    "4. Risk Simulator & Synthetic Demo",
])

# ==============================================================================
# TAB 1: EXECUTIVE & BUSINESS LAYER
# ==============================================================================
with tab_exec:
    if filtered_df.empty:
        st.warning("No customers match the current sidebar filters. Select at least one option per filter.")
    else:
        total_cust = len(filtered_df)
        churn_rate = filtered_df["churn"].mean() * 100
        high_risk_count = int(filtered_df["high_risk"].sum())
        monthly_at_risk = filtered_df["monthly_rev_at_risk"].sum()
        annual_at_risk = monthly_at_risk * 12

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers in Segment", f"{total_cust:,}")
        c2.metric("Observed Churn Rate", f"{churn_rate:.1f}%")
        c3.metric(f"High-Risk Accounts (≥{int(tuned_threshold*100)}%)", f"{high_risk_count:,}")
        c4.metric("Monthly Revenue at Risk", f"${monthly_at_risk:,.0f}", delta=f"${annual_at_risk:,.0f} / yr", delta_color="off")

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            seg_chart = (
                filtered_df.groupby("contract", as_index=False)
                .agg(churn_rate=("churn", lambda s: round(s.mean() * 100, 1)), rev_at_risk=("monthly_rev_at_risk", "sum"))
            )
            fig_contract = px.bar(
                seg_chart,
                x="contract",
                y="rev_at_risk",
                text="churn_rate",
                color="churn_rate",
                color_continuous_scale="RdYlGn_r",
                title="Monthly Revenue at Risk by Contract Type",
                labels={"contract": "Contract", "rev_at_risk": "Monthly Revenue at Risk ($)", "churn_rate": "Churn %"},
            )
            fig_contract.update_traces(texttemplate="%{text}% churn", textposition="outside")
            fig_contract.update_layout(height=360, margin=dict(t=45, b=20))
            st.plotly_chart(fig_contract, use_container_width=True)

        with col_right:
            ten_chart = (
                filtered_df.groupby("tenure_bucket", as_index=False)
                .agg(churn_rate=("churn", lambda s: round(s.mean() * 100, 1)), rev_at_risk=("monthly_rev_at_risk", "sum"))
            )
            ten_chart["tenure_bucket"] = pd.Categorical(ten_chart["tenure_bucket"], categories=tenures, ordered=True)
            ten_chart = ten_chart.sort_values("tenure_bucket")
            fig_tenure = px.line(
                ten_chart,
                x="tenure_bucket",
                y="churn_rate",
                markers=True,
                title="Churn Rate Decay Across Customer Tenure",
                labels={"tenure_bucket": "Tenure Bucket", "churn_rate": "Churn Rate (%)"},
            )
            fig_tenure.update_traces(line=dict(width=3, color="#EF553B"), marker=dict(size=9))
            fig_tenure.update_layout(height=360, margin=dict(t=45, b=20))
            st.plotly_chart(fig_tenure, use_container_width=True)

        st.subheader("Top 15 High-Risk Accounts by Monthly Revenue at Risk")
        st.caption("Ranked by Expected Monthly Revenue at Risk = Predicted Churn Probability × Monthly Charges.")
        top_accounts = (
            filtered_df.sort_values("monthly_rev_at_risk", ascending=False)
            .head(15)[[
                "customer_id", "contract", "tenure", "internet_service",
                "payment_method", "monthly_charges", "churn_prob", "monthly_rev_at_risk",
            ]]
            .copy()
        )
        top_accounts["churn_prob"] = (top_accounts["churn_prob"] * 100).round(1)
        top_accounts["monthly_rev_at_risk"] = top_accounts["monthly_rev_at_risk"].round(2)
        top_accounts.columns = [
            "Customer ID", "Contract", "Tenure (Mo)", "Internet Plan",
            "Payment Method", "Monthly Charge ($)", "Churn Prob (%)", "Monthly Rev at Risk ($)",
        ]
        st.dataframe(top_accounts, use_container_width=True, hide_index=True)

        # --- Retention Recommendations & Impact ---
        st.divider()
        st.subheader("Retention Strategy & Estimated Financial Impact")
        st.caption("Calculated directly from the full 7,043-customer IBM Telco benchmark rates.")

        m2m = scored_df[scored_df["contract"] == "Month-to-month"]
        one_yr = scored_df[scored_df["contract"] == "One year"]
        m2m_rate, one_yr_rate = m2m["churn"].mean(), one_yr["churn"].mean()
        m2m_save_annual = 0.10 * len(m2m) * (m2m_rate - one_yr_rate) * m2m["monthly_charges"].mean() * 12

        echeck = scored_df[scored_df["payment_method"] == "Electronic check"]
        autopay = scored_df[scored_df["payment_method"].str.contains("automatic", case=False)]
        ec_rate, ap_rate = echeck["churn"].mean(), autopay["churn"].mean()
        echeck_save_annual = 0.15 * len(echeck) * (ec_rate - ap_rate) * echeck["monthly_charges"].mean() * 12

        fiber_no_ts = scored_df[
            (scored_df["internet_service"] == "Fiber optic")
            & (scored_df["tech_support"] == "No")
            & (scored_df["tenure"] <= 12)
        ]
        fiber_with_ts = scored_df[
            (scored_df["internet_service"] == "Fiber optic")
            & (scored_df["tech_support"] == "Yes")
        ]
        fiber_delta = max(0.05, fiber_no_ts["churn"].mean() - fiber_with_ts["churn"].mean())
        fiber_save_annual = 0.15 * len(fiber_no_ts) * fiber_delta * fiber_no_ts["monthly_charges"].mean() * 12

        rec_df = pd.DataFrame([
            {
                "High-Risk Segment": f"Month-to-Month Contracts ({len(m2m):,} users, {m2m_rate*100:.1f}% churn)",
                "Recommended Intervention": "Offer 1-month free credit or 10% loyalty discount for upgrading to a 1-Year contract.",
                "Assumption": f"Convert 10% of segment ({int(len(m2m)*0.10):,} users) toward 1-Year churn baseline ({one_yr_rate*100:.1f}%).",
                "Est. Annual Revenue Saved": f"~${m2m_save_annual:,.0f} / yr",
            },
            {
                "High-Risk Segment": f"Electronic Check Payers ({len(echeck):,} users, {ec_rate*100:.1f}% churn)",
                "Recommended Intervention": "One-click Auto-Pay migration nudge paired with a $5/mo autopay billing credit.",
                "Assumption": f"Migrate 15% of segment ({int(len(echeck)*0.15):,} users) toward Auto-Pay churn ({ap_rate*100:.1f}%).",
                "Est. Annual Revenue Saved": f"~${echeck_save_annual:,.0f} / yr",
            },
            {
                "High-Risk Segment": f"0–12m Fiber Optic w/o Tech Support ({len(fiber_no_ts):,} users, {fiber_no_ts['churn'].mean()*100:.1f}% churn)",
                "Recommended Intervention": "Proactive Day-30 onboarding check-in + 90-day complimentary Tech Support & Online Security trial.",
                "Assumption": f"Engage 15% of segment ({int(len(fiber_no_ts)*0.15):,} users) to narrow gap toward supported Fiber users.",
                "Est. Annual Revenue Saved": f"~${fiber_save_annual:,.0f} / yr",
            },
        ])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        # --- A/B Test Design ---
        with st.expander("A/B Test Design: Annual Contract Conversion Offer (Month-to-Month Cohort)", expanded=False):
            n_per_arm = calc_ab_sample_size(m2m_rate, mde=0.05)
            st.markdown(
                f"""
                * **Target Population:** Active **Month-to-Month** subscribers flagged above the `{tuned_threshold:.2f}` risk threshold (baseline churn: **{m2m_rate*100:.1f}%**).
                * **Hypothesis:** Offering a **10% annual-lock discount** to high-risk Month-to-Month subscribers will reduce 90-day churn by at least **5.0 percentage points** (from {m2m_rate*100:.1f}% to {(m2m_rate-0.05)*100:.1f}%) while remaining net-positive on 12-month LTV.
                * **Experiment Split:** 50/50 randomized user-level split stratified by tenure bucket and internet plan (**Control:** standard billing vs. **Treatment:** in-app + email annual upgrade offer).
                * **Primary Metric:** 90-day retention rate (binary churn outcome).
                * **Guardrail Metrics:** Net Revenue per User (ARPU after discount), support ticket volume, and refund/cancellation inquiries.
                * **Required Sample Size:** **~{n_per_arm:,} users per arm** (**{n_per_arm*2:,} total**) for 80% statistical power ($1-\\beta=0.80$) at $\\alpha=0.05$ (two-sided z-test) to detect a 5pp absolute reduction.
                * **Run Duration:** 30-day enrollment window + 90-day post-exposure observation window.
                """
            )

# ==============================================================================
# TAB 2: MODEL EVALUATION & SHAP
# ==============================================================================
with tab_model:
    st.subheader("Stratified 5-Fold Cross-Validation Benchmark (Sorted by PR-AUC)")
    st.caption(
        "All preprocessing (`StandardScaler`, `OneHotEncoder`) and resampling (`SMOTE`) run strictly inside each training fold to prevent data leakage."
    )
    cv_display = cv_df[["Model", "PR_AUC", "ROC_AUC", "F1", "RECALL", "PRECISION", "ACCURACY"]]
    st.dataframe(cv_display, use_container_width=True, hide_index=True)

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    tuned = test_metrics["tuned"]
    default = test_metrics["default_050"]
    m1.metric("Holdout PR-AUC", f"{test_metrics['pr_auc']:.3f}")
    m2.metric("Holdout ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
    m3.metric(
        f"Recall (@ {tuned_threshold:.2f})",
        f"{tuned['recall']:.3f}",
        delta=f"{tuned['recall'] - default['recall']:+.3f} vs 0.50",
    )
    m4.metric(
        f"Precision (@ {tuned_threshold:.2f})",
        f"{tuned['precision']:.3f}",
        delta=f"{tuned['precision'] - default['precision']:+.3f} vs 0.50",
    )
    m5.metric(
        f"F1-Score (@ {tuned_threshold:.2f})",
        f"{tuned['f1']:.3f}",
        delta=f"{tuned['f1'] - default['f1']:+.3f} vs 0.50",
    )

    col_cm, col_th = st.columns(2)

    with col_cm:
        cm = test_metrics["confusion_matrix_tuned"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["Predicted: Stay", "Predicted: Churn"],
                y=["Actual: Stay", "Actual: Churn"],
                text=[[f"TN: {cm['tn']}", f"FP: {cm['fp']}"], [f"FN: {cm['fn']}", f"TP: {cm['tp']}"]],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=f"Holdout Test Confusion Matrix (Threshold = {tuned_threshold:.2f})",
            height=360,
            margin=dict(t=45, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_th:
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["recall"], mode="lines+markers", name="Recall"))
        fig_th.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["precision"], mode="lines+markers", name="Precision"))
        fig_th.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["f1"], mode="lines+markers", name="F1-Score"))
        fig_th.add_vline(x=tuned_threshold, line_dash="dash", line_color="orange", annotation_text=f"Tuned ({tuned_threshold:.2f})")
        fig_th.update_layout(
            title="Decision Threshold Trade-off (Out-of-Fold)",
            xaxis_title="Probability Cutoff",
            yaxis_title="Score",
            height=360,
            margin=dict(t=45, b=20),
        )
        st.plotly_chart(fig_th, use_container_width=True)

    st.info(
        f"**Why lower the threshold from 0.50 to {tuned_threshold:.2f}?** "
        "At 0.50, the model plays it safe and misses too many silent churners. Lowering the alarm threshold catches "
        "far more real churners (higher **Recall**) in exchange for more false alarms (lower **Precision**). "
        "In subscription retention, sending a low-cost outreach email or contract-upgrade offer to a happy user "
        "is far cheaper than losing months of recurring revenue from a missed churner."
    )

    st.divider()
    st.subheader("Why Customers Churn — Global SHAP Feature Importance (Top 12)")
    top_shap = shap_df.head(12).iloc[::-1]
    fig_shap = px.bar(
        top_shap,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Tealgrn",
        labels={"mean_abs_shap": "Mean |SHAP Value| (Impact on Churn Log-Odds)", "feature": "Feature"},
        title="Top Global Churn Drivers (SHAP on Unseen Test Split)",
    )
    fig_shap.update_layout(height=420, showlegend=False, coloraxis_showscale=False, margin=dict(t=45, b=20))
    st.plotly_chart(fig_shap, use_container_width=True)

# ==============================================================================
# TAB 3: SQL INSIGHTS
# ==============================================================================
with tab_sql:
    st.subheader("MySQL Analytical Views (CTEs & Window Functions)")
    st.caption(
        "Built with `WITH` CTEs, `SUM() OVER()`, `RANK() OVER()`, and `DENSE_RANK() OVER()` in `sql/analytics_views.sql`."
    )

    c_sql1, c_sql2 = st.columns(2)
    with c_sql1:
        st.markdown("**1. `v_churn_by_tenure`** — Churn Share & Risk Rank by Lifecycle Stage")
        st.dataframe(sql_views["v_churn_by_tenure"], use_container_width=True, hide_index=True)

    with c_sql2:
        st.markdown("**2. `v_churn_by_payment`** — Payment Method vs. Company Churn Average")
        st.dataframe(sql_views["v_churn_by_payment"], use_container_width=True, hide_index=True)

    st.markdown("**3. `v_churn_by_segment`** — Top 15 Multi-Dimensional Risk Cohorts (Contract × Tenure × Plan)")
    st.dataframe(sql_views["v_churn_by_segment"].head(15), use_container_width=True, hide_index=True)

    if os.path.exists("sql/analytics_views.sql"):
        with st.expander("View Underlying SQL Queries (`sql/analytics_views.sql`)", expanded=False):
            with open("sql/analytics_views.sql", "r") as f:
                st.code(f.read(), language="sql")

# ==============================================================================
# TAB 4: RISK SIMULATOR & SYNTHETIC DEMO SANDBOX
# ==============================================================================
with tab_sim:
    st.subheader("Live Customer Churn Risk Simulator (IBM Telco Model)")
    st.caption(f"Runs real-time inference through the serialized `{test_metrics['best_model']}` pipeline (`artifacts/best_model.pkl`).")

    with st.form("telco_sim_form"):
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            sim_tenure = st.slider("Tenure (Months)", 0, 72, 6)
            sim_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0, step=1.0)
            sim_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with s2:
            sim_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            sim_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            sim_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            sim_payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with s3:
            sim_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            sim_partner = st.selectbox("Has Partner", ["No", "Yes"])
            sim_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            sim_streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    sim_row = pd.DataFrame([{
        "gender": "Female",
        "senior_citizen": sim_senior,
        "partner": sim_partner,
        "dependents": sim_dependents,
        "tenure": sim_tenure,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": sim_internet,
        "online_security": sim_security,
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": sim_tech,
        "streaming_tv": sim_streaming_tv,
        "streaming_movies": "No",
        "contract": sim_contract,
        "paperless_billing": sim_paperless,
        "payment_method": sim_payment,
        "monthly_charges": float(sim_monthly),
        "total_charges": float(sim_monthly * sim_tenure),
    }])

    prob = float(model.predict_proba(sim_row[FEATURE_COLS])[0, 1])
    rev_risk_mo = prob * sim_monthly
    is_high_risk = prob >= tuned_threshold

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Churn Probability", f"{prob*100:.1f}%")
    r2.metric("Risk Classification", "HIGH RISK" if is_high_risk else "LOW RISK", delta=f"Cutoff: {tuned_threshold*100:.0f}%", delta_color="off")
    r3.metric("Expected Revenue at Risk", f"${rev_risk_mo:.2f} / mo", delta=f"${rev_risk_mo*12:.0f} / yr", delta_color="off")

    if is_high_risk:
        st.error("Recommended Action: Trigger proactive retention outreach (annual contract upgrade credit or complimentary Tech Support trial).")
    else:
        st.success("Account is below the tuned churn risk threshold—standard lifecycle nurture recommended.")

    st.divider()
    st.subheader("Synthetic Demo Data Sandbox (Faker Archive)")
    st.warning(
        "**SYNTHETIC DEMO DATA ONLY:** The records below come from `synthetic_demo/synthetic_demo_customers.csv` "
        "(rule-based Faker records preserved strictly for UI stress-testing). They are **never** used in SQL analytics views, "
        "model training, or reported benchmark metrics."
    )
    synth_path = "synthetic_demo/synthetic_demo_customers.csv"
    if os.path.exists(synth_path):
        synth_df = pd.read_csv(synth_path)
        st.caption(f"Loaded {len(synth_df):,} labeled synthetic demo records from `{synth_path}`.")
        st.dataframe(synth_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Synthetic demo CSV not found in `synthetic_demo/`.")