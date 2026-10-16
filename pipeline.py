"""
End-to-end pipeline:
  1. Clean raw orders
  2. Build customer features as of a CUTOFF date (no data leakage)
  3. RFM segmentation + K-Means clustering
  4. Churn model (label = no purchase in the HORIZON days AFTER the cutoff)
  5. Risk-adjusted CLV
  6. Save scores, model and all README images

Run:  python src/pipeline.py   (from the project root)
"""
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dashboards

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "orders.csv"
PROCESSED = ROOT / "data" / "processed"
IMAGES, MODELS, REPORTS = ROOT / "screenshots", ROOT / "models", ROOT / "reports"
for p in (PROCESSED, IMAGES, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-07-01")   # features use data up to here
HORIZON = 120                          # churn = no order in the next 120 days
SEED = 42
FEATURES = ["recency", "frequency", "monetary", "aov", "tenure_days",
            "avg_gap_days", "orders_last_90d", "n_categories",
            "discount_rate", "return_rate", "avg_review"]
PALETTE = {"primary": "#2F5BEA", "green": "#1FA971", "orange": "#F59E0B",
           "red": "#E5484D", "grey": "#6B7280"}


# ----------------------------------------------------------------------------
# 1. Cleaning
# ----------------------------------------------------------------------------
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["order_date"])
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    n1 = len(df)
    df = df.dropna(subset=["customer_id"])
    n_null = n1 - len(df)
    n2 = len(df)
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]   # drop cancellations
    n_cancel = n2 - len(df)

    df["customer_id"] = df["customer_id"].astype(int)
    df["total_amount"] = df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    df["net_amount"] = np.where(df["returned"] == 1, 0.0, df["total_amount"])  # returns earn nothing
    print(f"[clean] raw={n0:,} | duplicates={n_dup:,} | missing customer={n_null:,} "
          f"| cancellations={n_cancel:,} | clean={len(df):,}")
    return df


# ----------------------------------------------------------------------------
# 2. Features (built only from data <= cutoff) + churn label (data after cutoff)
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    hist = df[df["order_date"] <= CUTOFF].copy()
    hist["disc_flag"] = (hist["discount_pct"] > 0).astype(int)
    recent = hist[hist["order_date"] > CUTOFF - pd.Timedelta(days=90)]

    feat = hist.groupby("customer_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("net_amount", "sum"),
        n_categories=("category", "nunique"),
        discount_rate=("disc_flag", "mean"),
        return_rate=("returned", "mean"),
        avg_review=("review_score", "mean"),
        country=("country", "first"),
    )
    feat["recency"] = (CUTOFF - feat["last_order"]).dt.days
    feat["tenure_days"] = (CUTOFF - feat["first_order"]).dt.days
    feat["aov"] = feat["monetary"] / feat["frequency"]
    feat["avg_gap_days"] = np.where(feat["frequency"] > 1,
                                    feat["tenure_days"] / (feat["frequency"] - 1).clip(lower=1), 365)
    feat["orders_last_90d"] = recent.groupby("customer_id")["order_id"].nunique()
    feat["orders_last_90d"] = feat["orders_last_90d"].fillna(0)

    future = df[(df["order_date"] > CUTOFF) &
                (df["order_date"] <= CUTOFF + pd.Timedelta(days=HORIZON))]
    feat["churn"] = (~feat.index.isin(future["customer_id"].unique())).astype(int)
    return feat


# ----------------------------------------------------------------------------
# 3. RFM + clustering
# ----------------------------------------------------------------------------
def rfm_segments(feat: pd.DataFrame) -> pd.DataFrame:
    feat["R"] = pd.qcut(feat["recency"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    feat["F"] = pd.qcut(feat["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    feat["M"] = pd.qcut(feat["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def label(r):
        if r.R >= 4 and r.F >= 4:
            return "Champions"
        if r.R >= 3 and r.F >= 3:
            return "Loyal Customers"
        if r.R >= 4 and r.F <= 2:
            return "New / Promising"
        if r.R <= 2 and r.F >= 3:
            return "At Risk"
        if r.R <= 2 and r.F <= 2:
            return "Lost / Hibernating"
        return "Needs Attention"

    feat["segment"] = feat.apply(label, axis=1)
    return feat


def kmeans_clusters(feat: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X = StandardScaler().fit_transform(np.log1p(feat[["recency", "frequency", "monetary"]]))
    ks, inertia, sil = range(2, 9), [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_, sample_size=2000, random_state=SEED))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(ks), inertia, marker="o", color=PALETTE["primary"])
    ax[0].set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia")
    ax[1].plot(list(ks), sil, marker="o", color=PALETTE["green"])
    ax[1].set(title="Silhouette Score", xlabel="Number of clusters (k)", ylabel="Score")
    plt.tight_layout()
    plt.savefig(IMAGES / "elbow_silhouette.png", dpi=150)
    plt.close()

    feat["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    profile = feat.groupby("cluster").agg(customers=("recency", "size"), recency=("recency", "mean"),
                                          frequency=("frequency", "mean"), monetary=("monetary", "mean"),
                                          churn_rate=("churn", "mean")).round(2)
    profile.to_csv(REPORTS / "cluster_profile.csv")
    print("[cluster profile]\n", profile)
    return feat


# ----------------------------------------------------------------------------
# 4. Churn model
# ----------------------------------------------------------------------------
def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def train_and_evaluate(feat: pd.DataFrame):
    X, y = feat[FEATURES], feat["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    rows, fitted = [], {}
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred),
                     "F1-Score": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = model
    results = pd.DataFrame(rows).round(3).set_index("Model")
    results.to_csv(REPORTS / "model_comparison.csv")
    print("[models]\n", results)

    best_name = results["ROC-AUC"].idxmax()
    best = fitted[best_name]
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(IMAGES / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curves (all models)
    plt.figure(figsize=(5.5, 4.5))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC {results.loc[name, 'ROC-AUC']:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES / "roc_curve.png", dpi=150)
    plt.close()

    # permutation feature importance (model-agnostic)
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    imp_df = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 4.5))
    imp_df.plot.barh(color=PALETTE["primary"])
    plt.title(f"What drives churn? Permutation importance ({best_name})")
    plt.xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.tight_layout()
    plt.savefig(IMAGES / "feature_importance.png", dpi=150)
    plt.close()

    # optional SHAP plot (only if `shap` is installed and the model is tree-based)
    try:
        import shap
        if best_name != "Logistic Regression":
            sv = shap.TreeExplainer(best).shap_values(X_te)
            sv = sv[1] if isinstance(sv, list) else sv
            shap.summary_plot(sv, X_te, show=False)
            plt.tight_layout()
            plt.savefig(IMAGES / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("[shap] saved images/shap_summary.png")
    except Exception as exc:  # shap missing or incompatible: skip quietly
        print(f"[shap] skipped ({type(exc).__name__})")

    # out-of-fold probabilities for every customer (honest scores for the dashboard)
    oof = cross_val_predict(make_models()[best_name], X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]

    # final deployable model = best model refit on all data
    final = make_models()[best_name].fit(X, y)
    joblib.dump({"model": final, "features": FEATURES, "model_name": best_name},
                MODELS / "churn_model.pkl")
    return best_name, results, oof


# ----------------------------------------------------------------------------
# 5. CLV (simple, transparent heuristic)
# ----------------------------------------------------------------------------
def add_clv(feat: pd.DataFrame) -> pd.DataFrame:
    orders_per_year = feat["frequency"] / feat["tenure_days"].clip(lower=60) * 365
    feat["expected_annual_value"] = feat["aov"] * orders_per_year
    feat["risk_adjusted_clv"] = feat["expected_annual_value"] * (1 - feat["churn_prob"])
    feat["value_at_risk"] = feat["expected_annual_value"] * feat["churn_prob"]
    feat["risk_level"] = pd.cut(feat["churn_prob"], [-0.01, 0.33, 0.66, 1.01],
                                labels=["Low", "Medium", "High"])
    return feat


# ----------------------------------------------------------------------------
# 6. Dashboard screenshots live in src/dashboards.py
# ----------------------------------------------------------------------------
def rfm_chart(feat: pd.DataFrame):
    s = feat.groupby("segment").agg(customers=("recency", "size"), avg_spend=("monetary", "mean"),
                                    churn=("churn", "mean")).sort_values("avg_spend")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, color, title in zip(ax, ["customers", "avg_spend", "churn"],
                                    [PALETTE["primary"], PALETTE["green"], PALETTE["red"]],
                                    ["Customers", "Average Spend", "Actual Churn Rate"]):
        a.barh(s.index, s[col], color=color)
        a.set_title(title)
        a.grid(False)
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([])
    ax[2].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.suptitle("RFM Segment Profile", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES / "rfm_segments.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
def export_bi_tables(orders: pd.DataFrame, feat: pd.DataFrame):
    """Tables ready to import into Power BI (star schema)."""
    fact = orders[["order_id", "customer_id", "order_date", "category", "quantity", "unit_price",
                   "discount_pct", "returned", "review_score", "country", "net_amount"]].copy()
    fact["net_amount"] = fact["net_amount"].round(2)
    fact.to_csv(PROCESSED / "fact_orders.csv", index=False)
    dates = pd.date_range(orders["order_date"].min().replace(day=1), orders["order_date"].max(), freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["year"], dim["month_no"] = dim["date"].dt.year, dim["date"].dt.month
    dim["month"], dim["quarter"] = dim["date"].dt.strftime("%b"), "Q" + dim["date"].dt.quarter.astype(str)
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    dim.to_csv(PROCESSED / "dim_date.csv", index=False)
    out = feat.reset_index()
    for c in ["first_order", "last_order"]:
        out[c] = out[c].dt.date
    for c in ["monetary", "aov", "avg_gap_days", "discount_rate", "return_rate", "avg_review", "churn_prob",
              "expected_annual_value", "risk_adjusted_clv", "value_at_risk"]:
        out[c] = out[c].round(3)
    out.to_csv(PROCESSED / "customer_scores.csv", index=False)


if __name__ == "__main__":
    orders = load_and_clean()
    features = build_features(orders)
    print(f"[features] {len(features):,} customers | churn rate = {features['churn'].mean():.1%}")
    features = rfm_segments(features)
    features = kmeans_clusters(features)
    best_name, results, oof = train_and_evaluate(features)
    features["churn_prob"] = oof
    features = add_clv(features)
    export_bi_tables(orders, features)
    rfm_chart(features)
    dashboards.draw_all(orders, features, IMAGES, CUTOFF, HORIZON, results.loc[best_name, "ROC-AUC"])
    print(f"\nDone. Best model: {best_name}. Screenshots saved to {IMAGES}")
