# Customer Intelligence & Churn Analytics - Project Report

> **Data note:** all numbers below come from the *synthetic sample dataset* shipped with this repo so the project is
> reproducible. The method is real; the figures illustrate the output. Replace `data/raw/orders.csv` with a real
> dataset, rerun `python src/pipeline.py` and `python src/make_report.py`, and this report updates itself.

## 1. Executive summary
- **1,699 customers** were scored. **51%** placed no order in the 120 days after the model cutoff (2025-07-01).
- The top 20% of customers generate about **59%** of net revenue.
- The best model (**Logistic Regression**) reaches **ROC-AUC 0.94**, **precision 0.94** and **recall 0.79** on held-out customers.
- Estimated **annual value at risk: 658,580**. **664 High-risk customers** (39% of customers) hold **54%** of it.
- **At Risk** is the highest-priority segment: 20% of customers but 43% of value at risk.

## 2. Business problem
Winning a new customer costs much more than keeping an existing one. The business needs to know who its valuable
customers are, who is about to leave, why, and what to do about it.

## 3. Data and cleaning
- 28,325 raw order rows (order, customer, date, category, quantity, price, discount, return flag, review score, country, payment type).
- Removed exact duplicates, rows without a customer ID, and cancelled orders (negative quantity).
- Net revenue = quantity x price x (1 - discount); returned orders count as zero revenue.
- Electronics is the largest category (40% of net revenue), followed by Sports (18%).

## 4. Method
1. **Features** (one row per customer) built only from orders on or before **2025-07-01**: recency, frequency, spend, average order value, tenure, average gap between orders, orders in the last 90 days, categories bought, discount share, return rate, average review.
2. **Churn label:** no order in the **120 days after** the cutoff. Features never see the label period, so there is no data leakage.
3. **Segmentation:** RFM quintile scores mapped to six named segments, cross-checked with K-Means (k = 4) on log-scaled recency, frequency and spend.
4. **Modeling:** Logistic Regression, Random Forest and Gradient Boosting; stratified 80/20 split; selection by ROC-AUC. Dashboard probabilities are **out-of-fold** (5-fold cross-validation), so no customer is scored by a model that saw their label.
5. **Value:** expected annual value = average order value x orders per year; value at risk = expected annual value x churn probability.

## 5. Model performance (held-out test set, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.871 | 0.944 | 0.791 | 0.861 | 0.938 |
| Random Forest | 0.859 | 0.925 | 0.785 | 0.849 | 0.933 |
| Gradient Boosting | 0.85 | 0.917 | 0.773 | 0.839 | 0.924 |

## 6. Segment findings
| Segment | Customers | Avg days since order | Avg spend | Actual churn | Value at risk | Share of total |
|---|---|---|---|---|---|---|
| At Risk | 332 | 230 | 1,014 | 94% | 281,961 | 43% |
| Loyal Customers | 319 | 34 | 977 | 32% | 121,935 | 19% |
| Champions | 368 | 9 | 1,798 | 10% | 99,125 | 15% |
| Lost / Hibernating | 348 | 291 | 195 | 94% | 77,852 | 12% |
| New / Promising | 191 | 11 | 224 | 15% | 42,261 | 6% |
| Needs Attention | 141 | 50 | 201 | 40% | 35,445 | 5% |

## 7. Recommendations
| Segment | Action | Value at risk |
|---|---|---|
| At Risk | Win-back priority: personal outreach, time-limited offer, feedback survey | 281,961 |
| Loyal Customers | Grow basket size: cross-sell, bundles, subscription offers | 121,935 |
| Champions | Reward and retain: loyalty tier, early access, referral programme | 99,125 |
| Lost / Hibernating | Low-cost reactivation only; stop spending if no response | 77,852 |
| New / Promising | Win the second order: onboarding emails, 30-day incentive | 42,261 |
| Needs Attention | Re-engage: personalised recommendations and reminders | 35,445 |

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
