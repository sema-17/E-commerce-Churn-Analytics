"""
Streamlit app: score customers for churn risk.
Run:  streamlit run app/streamlit_app.py
(Run `python src/pipeline.py` first so the model and scores exist.)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Customer Churn Intelligence", page_icon="🛒", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "models" / "churn_model.pkl")
    return bundle["model"], bundle["features"], bundle["model_name"]


model, FEATURES, MODEL_NAME = load_model()


def risk_level(p: float) -> str:
    return "Low" if p < 0.33 else "Medium" if p < 0.66 else "High"


ACTIONS = {
    "Low": "Keep engaged: loyalty points, new-arrival emails, referral invites.",
    "Medium": "Nudge now: personalised recommendations and a small incentive on the next order.",
    "High": "Win-back priority: personal outreach, time-limited offer, and a feedback request.",
}


def reasons(r: dict) -> list[str]:
    out = []
    if r["orders_last_90d"] == 0:
        out.append("No orders in the last 90 days")
    if r["recency"] > 2 * r["avg_gap_days"]:
        out.append("Silent for more than twice their usual gap between orders")
    if r["return_rate"] > 0.15:
        out.append("High return rate (possible dissatisfaction)")
    if r["avg_review"] < 3.5:
        out.append("Low average review score")
    if r["discount_rate"] > 0.6:
        out.append("Mostly buys on discount (price-sensitive)")
    if r["frequency"] <= 1:
        out.append("Only one purchase so far")
    return out or ["No strong warning signs detected"]


st.title("🛒 Customer Churn Intelligence")
st.caption(f"Model: {MODEL_NAME}  |  Churn = no purchase in the next 120 days")

tab1, tab2, tab3 = st.tabs(["Single customer", "Batch scoring (CSV)", "Portfolio view"])

# ------------------------------------------------------------------ single
with tab1:
    c1, c2, c3 = st.columns(3)
    recency = c1.slider("Days since last order", 0, 365, 45)
    frequency = c1.slider("Total orders", 1, 60, 5)
    aov = c1.slider("Average order value", 5, 500, 80)
    tenure = c2.slider("Customer age (days)", 1, 700, 300)
    gap = c2.slider("Avg days between orders", 1, 365, 60)
    last90 = c2.slider("Orders in last 90 days", 0, 20, 1)
    ncat = c3.slider("Categories bought", 1, 6, 2)
    disc = c3.slider("Share of orders with discount", 0.0, 1.0, 0.3)
    ret = c3.slider("Return rate", 0.0, 1.0, 0.05)
    review = c3.slider("Average review score", 1.0, 5.0, 4.2)

    row = {"recency": recency, "frequency": frequency, "monetary": aov * frequency, "aov": aov,
           "tenure_days": tenure, "avg_gap_days": gap, "orders_last_90d": last90,
           "n_categories": ncat, "discount_rate": disc, "return_rate": ret, "avg_review": review}
    prob = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[:, 1][0])
    lvl = risk_level(prob)

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Risk level", lvl)
    st.progress(prob)
    st.subheader("Why?")
    for r in reasons(row):
        st.write(f"- {r}")
    st.subheader("Suggested action")
    st.info(ACTIONS[lvl])

# ------------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with these columns:")
    st.code(", ".join(FEATURES))
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["churn_prob"] = model.predict_proba(data[FEATURES])[:, 1]
            data["risk_level"] = data["churn_prob"].apply(risk_level)
            st.dataframe(data.sort_values("churn_prob", ascending=False), use_container_width=True)
            st.download_button("Download scored file", data.to_csv(index=False), "scored_customers.csv")

# --------------------------------------------------------------- portfolio
with tab3:
    scores_path = ROOT / "data" / "processed" / "customer_scores.csv"
    if scores_path.exists():
        sc = pd.read_csv(scores_path)
        k1, k2, k3 = st.columns(3)
        k1.metric("Customers", f"{len(sc):,}")
        k2.metric("High-risk customers", f"{(sc['risk_level'] == 'High').sum():,}")
        k3.metric("Annual value at risk", f"{sc['value_at_risk'].sum():,.0f}")
        seg = sc.groupby("segment").agg(customers=("customer_id", "size"),
                                        avg_churn_prob=("churn_prob", "mean"),
                                        value_at_risk=("value_at_risk", "sum")).round(2)
        st.dataframe(seg.sort_values("value_at_risk", ascending=False), use_container_width=True)
        st.bar_chart(seg["value_at_risk"])
        st.subheader("Top 20 high-value customers at risk")
        top = sc[sc["risk_level"] == "High"].sort_values("expected_annual_value", ascending=False).head(20)
        st.dataframe(top[["customer_id", "segment", "churn_prob", "expected_annual_value"]],
                     use_container_width=True)
    else:
        st.warning("Run `python src/pipeline.py` first to create customer_scores.csv")
