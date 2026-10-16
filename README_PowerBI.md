# Power BI Build Guide

Build the 3-page **Customer Intelligence** dashboard in about 45-60 minutes using the files in this repo.
The Python-drawn previews in [`../screenshots/`](../screenshots) show the target layout.

> **Status:** `customer_intelligence.pbix` is **not** included - a Power BI file can only be created in Power BI Desktop.
> After you build it, save it as `powerbi/customer_intelligence.pbix`, take screenshots of your pages, and
> replace the three `dashboard_*.png` files in `screenshots/` (keep the same names so the README links keep working).

## 1. Import the data
Open **Power BI Desktop → Get data → Text/CSV** and load:

| File | Table name | Rows are... |
|------|-----------|-------------|
| `data/processed/fact_orders.csv` | `fact_orders` | one row per cleaned order |
| `data/processed/customer_scores.csv` | `customer_scores` | one row per customer (features, segment, churn probability, CLV) |
| `data/processed/dim_date.csv` | `dim_date` | one row per calendar day |

In **Power Query** check the data types: dates → *Date*, `customer_id` → *Whole number*, `churn_prob`, `net_amount`,
`value_at_risk` → *Decimal number*. Remove nothing - the files are already cleaned.

## 2. Model (star schema)
In **Model view** create these relationships (all *many-to-one*, single direction):

- `fact_orders[order_date]` → `dim_date[date]`
- `fact_orders[customer_id]` → `customer_scores[customer_id]`

Then select `dim_date` → **Mark as date table** → column `date`.

> Some orders belong to customers who first bought *after* the model cutoff (2025-07-01), so they have no row in
> `customer_scores`. That is expected: revenue visuals use `fact_orders`; churn visuals use `customer_scores`.

Sort `dim_date[month]` by `dim_date[month_no]`.

## 3. Measures
Create an empty table called `_Measures` (**Home → Enter data**), then paste each measure from
[`dax_measures.dax`](dax_measures.dax). Format: revenue and value measures as number with thousands separator,
churn and risk percentages as **0.0%**.

## 4. Theme
**View → Themes → Browse for themes** → select [`theme.json`](theme.json).

## 5. Pages

### Page 1 - Executive Overview
| Visual | Fields |
|--------|--------|
| 4 × **Card** | `Total Revenue`, `Scored Customers`, `Avg Order Value`, `Actual Churn Rate` |
| **Line chart** | X: `dim_date[year_month]`, Y: `Total Revenue` |
| **Bar chart** | Axis: `customer_scores[segment]`, Values: `Scored Customers` |
| **Column chart** | Axis: `fact_orders[category]`, Values: `Total Revenue` |
| **100% stacked bar** | Axis: `segment`, Legend: `risk_level`, Values: `Scored Customers` |
| **Bar chart** | Axis: `segment`, Values: `Annual Value at Risk` |
| **Slicers** | `dim_date[year]`, `fact_orders[country]` |

### Page 2 - Customer Segments
| Visual | Fields |
|--------|--------|
| **Scatter chart** | X: `frequency`, Y: `monetary`, Legend: `segment`, Details: `customer_id` (use log axis scale) |
| **Bar chart** | Axis: `segment`, Values: average of `recency` |
| **Donut chart** | Legend: `segment`, Values: `Scored Customers` |
| **Table** | Rows: `cluster`; values: customers, avg `recency`, avg `frequency`, avg `monetary`, `Actual Churn Rate` |
| **Slicer** | `customer_scores[country]` |

### Page 3 - Churn Risk
| Visual | Fields |
|--------|--------|
| 4 × **Card** | `High Risk Customers`, `Avg Churn Probability`, `Annual Value at Risk`, `Recoverable Value (15%)` |
| **Histogram** | Column chart with a bin group on `churn_prob` (bin size 0.05) |
| **Column chart** | Axis: `recency` (bin size 30), Values: `Actual Churn Rate` |
| **Column chart** | Axis: `country`, Values: `Actual Churn Rate` |
| **Table (Top N)** | Columns: `customer_id`, `segment`, `churn_prob`, `recency`, `expected_annual_value`; filter `risk_level = High`, Top 10 by `expected_annual_value` |
| **Slicers** | `segment`, `risk_level` |

## 6. Polish checklist
- Page size: 16:9, background `#F4F6FA`, white rounded cards, consistent fonts
- Add **tooltips** and a **drill-through** page from `segment` to the customer table
- Add a title and a subtitle text (`Subtitle Text` measure) on each page
- Turn on **Sync slicers** for country across pages
- Name pages and visuals clearly (*Selection pane*) so the file is easy to review

## 7. Publish and share
Save the `.pbix` into this folder, take a 1920×1080 screenshot of each page (**File → Export → PDF** also works),
and optionally publish to Power BI Service and add the link to the main README.
