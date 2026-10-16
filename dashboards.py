"""
Draws three dashboard pages as PNG screenshots (Python/matplotlib version of the BI dashboard).
Page 1: Executive Overview | Page 2: Customer Segments | Page 3: Churn Risk
The same layout is described step by step in powerbi/README_PowerBI.md so you can
rebuild it in Power BI and replace these screenshots with your own.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

C = {"blue": "#2F5BEA", "green": "#1FA971", "orange": "#F59E0B", "red": "#E5484D",
     "grey": "#6B7280", "ink": "#111827", "bg": "#F4F6FA"}
SEG_COLORS = {"Champions": "#1FA971", "Loyal Customers": "#2F5BEA", "New / Promising": "#06B6D4",
              "Needs Attention": "#F59E0B", "At Risk": "#E5484D", "Lost / Hibernating": "#6B7280"}


def _grid(title, subtitle, hr):
    fig = plt.figure(figsize=(17, 10.5), facecolor=C["bg"])
    gs = GridSpec(len(hr), 12, figure=fig, height_ratios=hr, hspace=0.55, wspace=2.6,
                  left=0.04, right=0.97, top=0.88, bottom=0.07)
    fig.text(0.04, 0.945, title, fontsize=22, fontweight="bold", color=C["ink"])
    fig.text(0.04, 0.915, subtitle, fontsize=11, color=C["grey"])
    return fig, gs


def _card(fig, spec, title):
    ax = fig.add_subplot(spec)
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=C["ink"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def _kpis(fig, gs, items):
    for i, (title, value, color) in enumerate(items):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_color("#E5E7EB")
        ax.add_patch(plt.Rectangle((0, 0), 0.025, 1, color=color, transform=ax.transAxes))
        ax.text(0.08, 0.62, value, fontsize=26, fontweight="bold", color=C["ink"], transform=ax.transAxes)
        ax.text(0.08, 0.22, title, fontsize=12, color=C["grey"], transform=ax.transAxes)


def _k(x, _=None):
    return f"{x/1000:.0f}k"


def _table(ax, df, col_widths=None, fontsize=10):
    ax.axis("off")
    h = min(0.86, 0.105 * (len(df) + 1))          # keep the table below the card title
    t = ax.table(cellText=df.values, colLabels=list(df.columns), cellLoc="center",
                 colWidths=col_widths, bbox=[0, 0.86 - h, 1, h])
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    for (r, _c), cell in t.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if r == 0:
            cell.set_facecolor("#EEF2FF")
            cell.set_text_props(fontweight="bold", color=C["ink"])


# ---------------------------------------------------------------- page 1
def page1(df, feat, out, cutoff, horizon):
    fig, gs = _grid("Customer Intelligence - Executive Overview",
                    f"Sample data | features up to {cutoff.date()} | churn = no purchase in next {horizon} days",
                    [0.5, 1, 1])
    _kpis(fig, gs, [("Net Revenue", f"{df['net_amount'].sum():,.0f}", C["blue"]),
                    ("Customers Scored", f"{len(feat):,}", C["green"]),
                    ("Avg Order Value", f"{feat['aov'].mean():,.1f}", C["orange"]),
                    ("Churn Rate", f"{feat['churn'].mean():.1%}", C["red"])])

    ax = _card(fig, gs[1, 0:8], "Monthly Net Revenue")
    m = df.set_index("order_date")["net_amount"].resample("MS").sum()
    ax.plot(m.index, m.values, color=C["blue"], lw=2.5, marker="o", ms=4)
    ax.fill_between(m.index, m.values, alpha=0.12, color=C["blue"])
    ax.axvline(cutoff, ls="--", color=C["red"], lw=1)
    ax.text(cutoff, ax.get_ylim()[1] * 0.95, "  model cutoff", color=C["red"], fontsize=9, va="top")
    ax.yaxis.set_major_formatter(_k)

    ax = _card(fig, gs[1, 9:12], "Customers by RFM Segment")
    seg = feat["segment"].value_counts().sort_values()
    ax.barh(seg.index, seg.values, color=[SEG_COLORS[s] for s in seg.index])
    for i, v in enumerate(seg.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    ax.grid(False)

    ax = _card(fig, gs[2, 0:3], "Revenue by Category")
    cat = df.groupby("category")["net_amount"].sum().sort_values(ascending=False)
    ax.bar(cat.index, cat.values, color=C["orange"])
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.yaxis.set_major_formatter(_k)

    ax = _card(fig, gs[2, 4:8], "Churn Risk Mix by Segment")
    mix = pd.crosstab(feat["segment"], feat["risk_level"], normalize="index")[["Low", "Medium", "High"]]
    mix = mix.sort_values("High")
    left = np.zeros(len(mix))
    for lvl, col in zip(mix.columns, [C["green"], C["orange"], C["red"]]):
        ax.barh(mix.index, mix[lvl], left=left, color=col, label=lvl)
        left += mix[lvl].values
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(False)

    ax = _card(fig, gs[2, 9:12], "Annual Value at Risk by Segment")
    var = feat.groupby("segment")["value_at_risk"].sum().sort_values()
    ax.barh(var.index, var.values, color=C["red"])
    ax.xaxis.set_major_formatter(_k)
    ax.grid(False)
    fig.savefig(out / "dashboard_1_executive_overview.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- page 2
def page2(feat, out, cutoff, horizon):
    fig, gs = _grid("Customer Intelligence - Customer Segments",
                    "RFM segments and K-Means clusters (sample data)", [1, 1])

    ax = _card(fig, gs[0, 0:6], "Frequency vs Spend (log scale, coloured by RFM segment)")
    for s, g in feat.groupby("segment"):
        ax.scatter(g["frequency"], g["monetary"].clip(lower=1), s=14, alpha=0.55,
                   color=SEG_COLORS[s], label=s)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Orders (frequency)"); ax.set_ylabel("Net spend (monetary)")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")

    ax = _card(fig, gs[0, 7:12], "Average Days Since Last Order")
    rec = feat.groupby("segment")["recency"].mean().sort_values()
    ax.barh(rec.index, rec.values, color=[SEG_COLORS[s] for s in rec.index])
    for i, v in enumerate(rec.values):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=9)
    ax.grid(False)

    ax = _card(fig, gs[1, 0:4], "Customer Share by Segment")
    share = feat["segment"].value_counts()
    ax.pie(share.values, labels=share.index, autopct="%1.0f%%", startangle=90, pctdistance=0.82,
           colors=[SEG_COLORS[s] for s in share.index], wedgeprops=dict(width=0.38, edgecolor="white"),
           textprops={"fontsize": 8})
    ax.set_aspect("equal")

    ax = _card(fig, gs[1, 5:12], "K-Means Cluster Profile")
    prof = feat.groupby("cluster").agg(customers=("recency", "size"), recency=("recency", "mean"),
                                       frequency=("frequency", "mean"), monetary=("monetary", "mean"),
                                       churn=("churn", "mean")).reset_index()
    prof["cluster"] = "Cluster " + prof["cluster"].astype(str)
    prof["customers"] = prof["customers"].map("{:,}".format)
    prof["recency"] = prof["recency"].map("{:.0f} d".format)
    prof["frequency"] = prof["frequency"].map("{:.1f}".format)
    prof["monetary"] = prof["monetary"].map("{:,.0f}".format)
    prof["churn"] = prof["churn"].map("{:.0%}".format)
    prof.columns = ["Cluster", "Customers", "Avg recency", "Avg orders", "Avg spend", "Churn rate"]
    _table(ax, prof, fontsize=11)
    fig.savefig(out / "dashboard_2_customer_segments.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- page 3
def page3(feat, out, cutoff, horizon, auc):
    fig, gs = _grid("Customer Intelligence - Churn Risk",
                    "Out-of-fold churn probabilities for every scored customer (sample data)", [0.5, 1, 1])
    _kpis(fig, gs, [("High-Risk Customers", f"{(feat['risk_level'] == 'High').sum():,}", C["red"]),
                    ("Avg Churn Probability", f"{feat['churn_prob'].mean():.1%}", C["orange"]),
                    ("Annual Value at Risk", f"{feat['value_at_risk'].sum():,.0f}", C["blue"]),
                    ("Model ROC-AUC", f"{auc:.2f}", C["green"])])

    ax = _card(fig, gs[1, 0:6], "Distribution of Churn Probability")
    ax.hist(feat["churn_prob"], bins=30, color=C["blue"], alpha=0.85)
    for x, lab in [(0.33, "Low | Medium"), (0.66, "Medium | High")]:
        ax.axvline(x, ls="--", color=C["red"], lw=1)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {lab}", color=C["red"], fontsize=9, va="top")
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Customers")

    ax = _card(fig, gs[1, 7:12], "Actual Churn Rate by Recency")
    b = pd.cut(feat["recency"], [-1, 30, 60, 90, 150, 365, 1000],
               labels=["0-30", "31-60", "61-90", "91-150", "151-365", "365+"])
    r = feat.groupby(b, observed=True)["churn"].mean()
    ax.bar(r.index.astype(str), r.values, color=C["red"])
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Days since last order")

    ax = _card(fig, gs[2, 0:4], "Churn Rate by Country")
    cr = feat.groupby("country")["churn"].mean().sort_values(ascending=False)
    ax.bar(cr.index, cr.values, color=C["orange"])
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    ax = _card(fig, gs[2, 5:12], "Top 8 High-Value Customers at High Risk")
    top = feat[feat["risk_level"] == "High"].sort_values("expected_annual_value", ascending=False).head(8)
    t = pd.DataFrame({"Customer": top.index.astype(str), "Segment": top["segment"].values,
                      "Churn prob.": (top["churn_prob"] * 100).round(0).astype(int).astype(str) + "%",
                      "Days since order": top["recency"].astype(int).values,
                      "Annual value": top["expected_annual_value"].round(0).astype(int).map("{:,}".format).values})
    _table(ax, t, fontsize=10)
    fig.savefig(out / "dashboard_3_churn_risk.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_all(orders, feat, out: Path, cutoff, horizon, auc):
    out.mkdir(parents=True, exist_ok=True)
    page1(orders, feat, out, cutoff, horizon)
    page2(feat, out, cutoff, horizon)
    page3(feat, out, cutoff, horizon, auc)
