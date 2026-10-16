"""
Writes reports/project_report.md using the numbers produced by src/pipeline.py,
so the report always matches the data. Run after the pipeline:

    python src/make_report.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sc = pd.read_csv(ROOT / "data" / "processed" / "customer_scores.csv")
fact = pd.read_csv(ROOT / "data" / "processed" / "fact_orders.csv", parse_dates=["order_date"])
models = pd.read_csv(ROOT / "reports" / "model_comparison.csv", index_col=0)
raw_rows = sum(1 for _ in open(ROOT / "data" / "raw" / "orders.csv")) - 1


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


# ----- numbers -----
n_cust = len(sc)
churn_rate = sc["churn"].mean()
best = models["ROC-AUC"].idxmax()
b = models.loc[best]
total_var = sc["value_at_risk"].sum()
high = sc[sc["risk_level"] == "High"]
top20 = sc["monetary"].sort_values(ascending=False).head(int(n_cust * 0.2)).sum() / sc["monetary"].sum()
cat_share = fact.groupby("category")["net_amount"].sum().pipe(lambda s: s / s.sum()).sort_values(ascending=False)

seg = sc.groupby("segment").agg(customers=("customer_id", "size"), avg_recency=("recency", "mean"),
                                avg_spend=("monetary", "mean"), churn_rate=("churn", "mean"),
                                value_at_risk=("value_at_risk", "sum"))
seg["share_of_var"] = seg["value_at_risk"] / seg["value_at_risk"].sum()
seg = seg.sort_values("value_at_risk", ascending=False)
seg_tbl = pd.DataFrame({
    "Customers": seg["customers"].map("{:,}".format),
    "Avg days since order": seg["avg_recency"].map("{:.0f}".format),
    "Avg spend": seg["avg_spend"].map("{:,.0f}".format),
    "Actual churn": seg["churn_rate"].map("{:.0%}".format),
    "Value at risk": seg["value_at_risk"].map("{:,.0f}".format),
    "Share of total": seg["share_of_var"].map("{:.0%}".format)}, index=seg.index)
seg_tbl.index.name = "Segment"

mod_tbl = models.round(3).astype(str)
mod_tbl.index.name = "Model"

actions = {
    "At Risk": "Win-back priority: personal outreach, time-limited offer, feedback survey",
    "Loyal Customers": "Grow basket size: cross-sell, bundles, subscription offers",
    "Champions": "Reward and retain: loyalty tier, early access, referral programme",
    "Lost / Hibernating": "Low-cost reactivation only; stop spending if no response",
    "New / Promising": "Win the second order: onboarding emails, 30-day incentive",
    "Needs Attention": "Re-engage: personalised recommendations and reminders",
}
rec_tbl = pd.DataFrame({"Action": [actions[s] for s in seg.index],
                        "Value at risk": seg["value_at_risk"].map("{:,.0f}".format).values}, index=seg.index)
rec_tbl.index.name = "Segment"

report = f"""# Customer Intelligence & Churn Analytics - Project Report

> **Data note:** all numbers below come from the *synthetic sample dataset* shipped with this repo so the project is
> reproducible. The method is real; the figures illustrate the output. Replace `data/raw/orders.csv` with a real
> dataset, rerun `python src/pipeline.py` and `python src/make_report.py`, and this report updates itself.

## 1. Executive summary
- **{n_cust:,} customers** were scored. **{churn_rate:.0%}** placed no order in the 120 days after the model cutoff (2025-07-01).
- The top 20% of customers generate about **{top20:.0%}** of net revenue.
- The best model (**{best}**) reaches **ROC-AUC {b['ROC-AUC']:.2f}**, **precision {b['Precision']:.2f}** and **recall {b['Recall']:.2f}** on held-out customers.
- Estimated **annual value at risk: {total_var:,.0f}**. **{len(high):,} High-risk customers** ({len(high)/n_cust:.0%} of customers) hold **{high['value_at_risk'].sum()/total_var:.0%}** of it.
- **At Risk** is the highest-priority segment: {seg.loc['At Risk', 'customers']/n_cust:.0%} of customers but {seg.loc['At Risk', 'share_of_var']:.0%} of value at risk.

## 2. Business problem
Winning a new customer costs much more than keeping an existing one. The business needs to know who its valuable
customers are, who is about to leave, why, and what to do about it.

## 3. Data and cleaning
- {raw_rows:,} raw order rows (order, customer, date, category, quantity, price, discount, return flag, review score, country, payment type).
- Removed exact duplicates, rows without a customer ID, and cancelled orders (negative quantity).
- Net revenue = quantity x price x (1 - discount); returned orders count as zero revenue.
- Electronics is the largest category ({cat_share.iloc[0]:.0%} of net revenue), followed by {cat_share.index[1]} ({cat_share.iloc[1]:.0%}).

## 4. Method
1. **Features** (one row per customer) built only from orders on or before **2025-07-01**: recency, frequency, spend, average order value, tenure, average gap between orders, orders in the last 90 days, categories bought, discount share, return rate, average review.
2. **Churn label:** no order in the **120 days after** the cutoff. Features never see the label period, so there is no data leakage.
3. **Segmentation:** RFM quintile scores mapped to six named segments, cross-checked with K-Means (k = 4) on log-scaled recency, frequency and spend.
4. **Modeling:** Logistic Regression, Random Forest and Gradient Boosting; stratified 80/20 split; selection by ROC-AUC. Dashboard probabilities are **out-of-fold** (5-fold cross-validation), so no customer is scored by a model that saw their label.
5. **Value:** expected annual value = average order value x orders per year; value at risk = expected annual value x churn probability.

## 5. Model performance (held-out test set, threshold 0.5)
{md_table(mod_tbl)}

## 6. Segment findings
{md_table(seg_tbl)}

## 7. Recommendations
{md_table(rec_tbl)}

Contact only **High-risk, high-value** customers first. Lowering the decision threshold below 0.5 catches more churners
at the cost of more false alarms (see notebook 03).

## 8. Limitations
- **Synthetic data:** results show the workflow, not real customer behaviour.
- **Churn definition:** "no order in 120 days" is a business choice; the right window depends on normal purchase cycles.
- **Value estimates** are simple heuristics, not probabilistic CLV.
- **Correlation, not causation:** the model ranks risk; it does not prove a campaign will work. Test with an A/B experiment.
- Model scores should be refreshed regularly as behaviour changes.

## 9. Next steps
Probabilistic CLV (BG/NBD + Gamma-Gamma), SHAP explanations, a retention-campaign A/B test, scheduled monthly scoring,
and deploying the Streamlit app.
"""
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "project_report.md").write_text(report, encoding="utf-8")
print("Wrote reports/project_report.md")
