import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import mysql.connector
import joblib
import plotly.express as px

load_dotenv()

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

# Load trained model and exact feature columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@st.cache_data
def load_data():
    if os.getenv("DB_HOST"):
        try:
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                connection_timeout=3
            )
            query = "SELECT id, name, plan, support_tickets, days_inactive, churned FROM ml_customers"
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception:
            pass
    return pd.read_csv("customers_backup.csv")

df = load_data()

# --- Header & KPI Row ---
st.title("StayMetric — Customer Churn Intelligence")
st.caption("Live analytics backed by MySQL + Random Forest predictive scoring")

total_customers = len(df)
churn_rate = df["churned"].mean() * 100
high_risk_count = len(df[(df["support_tickets"] > 4) & (df["days_inactive"] > 20)])
avg_tickets = df["support_tickets"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
c3.metric("High-Risk Segment", f"{high_risk_count:,}")
c4.metric("Avg. Support Tickets", f"{avg_tickets:.2f}")

st.divider()

# --- Analytical Charts ---
col_left, col_right = st.columns(2)

with col_left:
    plan_churn = df.groupby("plan", as_index=False)["churned"].mean()
    plan_churn["churn_pct"] = plan_churn["churned"] * 100
    fig_plan = px.bar(
        plan_churn,
        x="plan",
        y="churn_pct",
        title="Churn Rate by Subscription Plan (%)",
        labels={"plan": "Plan Tier", "churn_pct": "Churn Rate (%)"},
        color="plan"
    )
    st.plotly_chart(fig_plan, width="stretch")

with col_right:
    ticket_churn = df.groupby("support_tickets", as_index=False)["churned"].mean()
    ticket_churn["churn_pct"] = ticket_churn["churned"] * 100
    fig_tickets = px.line(
        ticket_churn,
        x="support_tickets",
        y="churn_pct",
        markers=True,
        title="Churn Rate vs. Support Ticket Volume",
        labels={"support_tickets": "Support Tickets", "churn_pct": "Churn Rate (%)"}
    )
    st.plotly_chart(fig_tickets, width="stretch")

st.divider()

# --- Live Churn Risk Simulator ---
st.subheader("Live Customer Churn Predictor")

s1, s2, s3 = st.columns(3)
with s1:
    selected_plan = st.selectbox("Subscription Plan", ["Basic", "Pro", "Enterprise"])
with s2:
    tickets = st.slider("Support Tickets (Last 30 Days)", 0, 10, 3)
with s3:
    days_inactive = st.slider("Days Since Last Login", 1, 45, 15)

if st.button("Predict Churn Risk", type="primary"):
    input_df = pd.DataFrame([{
        "plan": selected_plan,
        "support_tickets": tickets,
        "days_inactive": days_inactive
    }])

    input_encoded = pd.get_dummies(input_df, columns=["plan"])
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(input_aligned)[0][1] * 100
    prediction = model.predict(input_aligned)[0]

    if prediction == 1:
        st.error(f"High Churn Risk: **{prob:.1f}% probability of cancellation.** Recommend immediate outreach.")
    else:
        st.success(f"Low Churn Risk: **{prob:.1f}% probability of cancellation.** Account appears stable.")