# 🛒 AI-Powered E-commerce Customer Intelligence & Churn Analytics Platform

An end-to-end data analytics project: clean raw e-commerce orders with **SQL and Python**, segment customers with **RFM and K-Means**, predict **customer churn** with machine learning, estimate the **revenue at risk**, and deliver the results through a **Power BI dashboard** and a **Streamlit app**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-orange)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-green)
![Power BI](https://img.shields.io/badge/BI-Power%20BI-yellow)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![Executive Overview](screenshots/dashboard_1_executive_overview.png)

> **About the data:** this repo ships a small **synthetic** order dataset (about 1.8 MB, 28k rows, 2,200 customers)
> so everyone can run the project in minutes. The numbers below demonstrate the workflow and are **not real business
> results**. Swap in a real dataset (Online Retail II, Olist) and rerun - see [Using your own data](#-using-your-own-data).

---

## 📌 Table of Contents
1. [Business problem](#-business-problem)
2. [What this project delivers](#-what-this-project-delivers)
3. [Repository structure](#-repository-structure)
4. [Tech stack](#-tech-stack)
5. [Methodology](#-methodology)
6. [Results (sample data)](#-results-sample-data)
7. [Dashboard screenshots](#-dashboard-screenshots)
8. [Business recommendations](#-business-recommendations)
9. [How to run](#-how-to-run)
10. [Using your own data](#-using-your-own-data)
11. [Skills demonstrated](#-skills-demonstrated)
12. [Limitations & next steps](#-limitations--next-steps)
13. [Author](#-author)

---

## 🎯 Business problem
Winning a new customer costs far more than keeping an existing one. An online retailer needs to know:

1. Who are our most valuable customers?
2. Which customers are about to leave?
3. How much revenue is at risk?
4. What should we do to keep them?

## ✅ What this project delivers
| Deliverable | Where |
|---|---|
| Cleaned, analysis-ready data | `data/processed/` |
| SQL for cleaning, RFM and business questions | `sql/customer_metrics.sql` |
| 4 executed notebooks (cleaning/EDA, RFM + clustering, churn model, CLV + recommendations) | `notebooks/` |
| Reproducible pipeline (data → model → charts) | `src/pipeline.py` |
| Churn prediction model | `models/churn_model.pkl` |
| Power BI build guide, DAX measures and theme | `powerbi/` |
| Interactive churn-scoring app | `app/streamlit_app.py` |
| Written business report | `reports/project_report.md` |
| Dashboard and chart screenshots | `screenshots/` |

## 📁 Repository structure
```
ecommerce-churn-analytics/
├── data/
│   ├── raw/orders.csv                  # synthetic sample orders (1.8 MB)
│   └── processed/
│       ├── fact_orders.csv             # cleaned orders (Power BI fact table)
│       ├── customer_scores.csv         # features, segment, churn probability, CLV
│       └── dim_date.csv                # calendar table
├── notebooks/
│   ├── 01_data_cleaning_and_eda.ipynb
│   ├── 02_rfm_segmentation_clustering.ipynb
│   ├── 03_churn_modeling.ipynb
│   └── 04_clv_and_recommendations.ipynb
├── sql/customer_metrics.sql            # PostgreSQL views, RFM with NTILE, business queries
├── src/
│   ├── generate_sample_data.py         # builds the synthetic dataset
│   ├── pipeline.py                     # clean → features → RFM → clustering → model → CLV
│   ├── dashboards.py                   # draws the dashboard screenshots
│   └── make_report.py                  # writes reports/project_report.md from results
├── app/streamlit_app.py                # churn scoring app
├── powerbi/
│   ├── README_PowerBI.md               # step-by-step dashboard build guide
│   ├── dax_measures.dax                # all DAX measures
│   └── theme.json                      # Power BI theme
├── models/churn_model.pkl
├── reports/                            # project_report.md, model_comparison.csv, cluster_profile.csv
├── screenshots/                        # dashboard pages and model charts
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🧰 Tech stack
| Area | Tools |
|---|---|
| Data handling | Python (pandas, NumPy), SQL (PostgreSQL) |
| Visualization | Matplotlib, Seaborn, Power BI |
| Machine learning | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, K-Means) |
| App | Streamlit |
| Workflow | Jupyter, Git & GitHub |

## 🔬 Methodology
1. **Cleaning:** removed duplicates, rows without a customer ID and cancelled orders; net revenue = quantity × price × (1 − discount), with returned orders counted as zero.
2. **Leak-free design:** customer features are built only from orders **up to 2025-07-01**. The churn label is **no purchase in the next 120 days** after that date.
3. **Features:** recency, frequency, spend, average order value, tenure, average gap between orders, orders in the last 90 days, categories bought, discount share, return rate, review score.
4. **Segmentation:** RFM quintile scores → six named segments, cross-checked with K-Means (k = 4, elbow + silhouette).
5. **Modeling:** three models compared on a stratified 80/20 split; best chosen by ROC-AUC. Scores shown in the dashboard are **out-of-fold** (5-fold CV).
6. **Value at risk:** expected annual value (average order value × orders per year) × churn probability. A transparent heuristic, not a probabilistic CLV model.

## 📈 Results (sample data)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.871 | 0.944 | 0.791 | 0.861 | **0.938** |
| Random Forest | 0.859 | 0.925 | 0.785 | 0.849 | 0.933 |
| Gradient Boosting | 0.850 | 0.917 | 0.773 | 0.839 | 0.924 |

- 1,699 customers scored; **51%** did not return in the 120-day window.
- The top 20% of customers generate about **59%** of net revenue.
- **Recency** is by far the strongest churn driver, followed by orders in the last 90 days.
- Estimated annual **value at risk: 658,580**; the 664 High-risk customers (39%) hold 54% of it.
- **At Risk** customers are 20% of the base but 43% of the value at risk.

Full write-up: [`reports/project_report.md`](reports/project_report.md)

| Confusion matrix | ROC curves |
|---|---|
| ![Confusion matrix](screenshots/confusion_matrix.png) | ![ROC](screenshots/roc_curve.png) |

![Feature importance](screenshots/feature_importance.png)

## 🖥️ Dashboard screenshots
The three pages below are drawn with Python from the sample data. The matching **Power BI** build (data model, DAX
measures, visuals, theme) is documented in [`powerbi/README_PowerBI.md`](powerbi/README_PowerBI.md).

**Page 2 - Customer segments**
![Customer segments](screenshots/dashboard_2_customer_segments.png)

**Page 3 - Churn risk**
![Churn risk](screenshots/dashboard_3_churn_risk.png)

## 🎯 Business recommendations
| Segment | Action |
|---|---|
| **At Risk** | Win-back priority: personal outreach, time-limited offer, feedback survey |
| **Loyal Customers** | Grow basket size: cross-sell, bundles, subscription offers |
| **Champions** | Reward and retain: loyalty tier, early access, referral programme |
| **New / Promising** | Win the second order: onboarding emails, 30-day incentive |
| **Needs Attention** | Re-engage with personalised recommendations |
| **Lost / Hibernating** | Low-cost reactivation only; stop spending if there is no response |

Contact **high-risk, high-value** customers first, and run an A/B test to confirm that campaigns actually reduce churn.

## 🚀 How to run
```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/ecommerce-churn-analytics.git
cd ecommerce-churn-analytics

# 2. Create an environment and install dependencies
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full pipeline (cleans data, trains models, saves scores and screenshots)
python src/pipeline.py
python src/make_report.py           # regenerates reports/project_report.md

# 4. Explore the notebooks
jupyter notebook notebooks/

# 5. Launch the churn-scoring app
streamlit run app/streamlit_app.py
```
Optional: regenerate the sample data with `python src/generate_sample_data.py`.
SQL: load `data/raw/orders.csv` into PostgreSQL and run `sql/customer_metrics.sql`.

## 🔁 Using your own data
1. Download a real dataset, e.g. **Online Retail II** (UCI) or **Olist Brazilian E-commerce** (Kaggle). Check its licence before redistributing it.
2. Map its columns to this schema and save it as `data/raw/orders.csv`:
   `order_id, customer_id, order_date, category, quantity, unit_price, discount_pct, returned, review_score, country, payment_type`
   (Columns you do not have can be filled with defaults, e.g. `discount_pct = 0`.)
3. Adjust `CUTOFF` and `HORIZON` at the top of `src/pipeline.py` to fit your date range.
4. Rerun the pipeline, the report script and the notebooks.

## 💼 Skills demonstrated
Data cleaning · SQL (CTEs, window functions, views) · exploratory analysis · customer segmentation (RFM, K-Means) ·
churn modeling and evaluation · avoiding data leakage · feature importance · KPI design · dashboarding (Power BI, DAX) ·
business storytelling · reproducible project structure.

## ⚠️ Limitations & next steps
- Results come from **synthetic data**; validate on a real dataset.
- Churn depends on the chosen 120-day window.
- Value estimates are simple heuristics; try BG/NBD + Gamma-Gamma CLV.
- Add SHAP explanations, an A/B test of retention campaigns, scheduled monthly scoring and a deployed Streamlit app.

## 👤 Author
**Your Name**
- LinkedIn: [your-linkedin-url](https://linkedin.com/in/your-profile)
- GitHub: [your-github-username](https://github.com/your-username)
- Email: your.email@example.com

## 📄 License
MIT - see [LICENSE](LICENSE).
