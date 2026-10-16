-- PostgreSQL. Load data/raw/orders.csv into the `orders` table first.

CREATE TABLE IF NOT EXISTS orders (
    order_id      BIGINT,
    customer_id   NUMERIC,
    order_date    DATE,
    category      TEXT,
    quantity      INT,
    unit_price    NUMERIC(10,2),
    discount_pct  INT,
    returned      INT,
    review_score  INT,
    country       TEXT,
    payment_type  TEXT
);
-- \copy orders FROM 'data/raw/orders.csv' CSV HEADER;

-- 1) Clean view: remove duplicates, missing customers and cancellations
CREATE OR REPLACE VIEW clean_orders AS
SELECT DISTINCT
       order_id, customer_id::INT AS customer_id, order_date, category, quantity,
       unit_price, discount_pct, returned, review_score, country,
       quantity * unit_price * (1 - discount_pct / 100.0)                       AS total_amount,
       CASE WHEN returned = 1 THEN 0
            ELSE quantity * unit_price * (1 - discount_pct / 100.0) END         AS net_amount
FROM orders
WHERE customer_id IS NOT NULL AND quantity > 0 AND unit_price > 0;

-- 2) Customer metrics as of a cutoff date (no future data = no leakage)
CREATE OR REPLACE VIEW customer_metrics AS
SELECT customer_id,
       DATE '2025-07-01' - MAX(order_date)         AS recency,
       COUNT(DISTINCT order_id)                    AS frequency,
       SUM(net_amount)                             AS monetary,
       SUM(net_amount) / COUNT(DISTINCT order_id)  AS aov,
       AVG(returned::NUMERIC)                      AS return_rate,
       AVG(review_score)                           AS avg_review
FROM clean_orders
WHERE order_date <= DATE '2025-07-01'
GROUP BY customer_id;

-- 3) RFM scores (1-5) and segments
CREATE OR REPLACE VIEW rfm_segments AS
WITH scored AS (
    SELECT *,
           NTILE(5) OVER (ORDER BY recency DESC)  AS r_score,
           NTILE(5) OVER (ORDER BY frequency)     AS f_score,
           NTILE(5) OVER (ORDER BY monetary)      AS m_score
    FROM customer_metrics
)
SELECT *,
       CASE WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'New / Promising'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost / Hibernating'
            ELSE 'Needs Attention' END AS segment
FROM scored;

-- 4) Business questions
-- Revenue by month
SELECT DATE_TRUNC('month', order_date) AS month, SUM(net_amount) AS revenue
FROM clean_orders GROUP BY 1 ORDER BY 1;

-- Segment summary
SELECT segment, COUNT(*) AS customers, ROUND(AVG(monetary),2) AS avg_spend
FROM rfm_segments GROUP BY segment ORDER BY avg_spend DESC;

-- Share of revenue from the top 20% of customers
WITH ranked AS (
    SELECT customer_id, monetary, NTILE(5) OVER (ORDER BY monetary DESC) AS quintile
    FROM customer_metrics
)
SELECT ROUND(100 * SUM(monetary) FILTER (WHERE quintile = 1) / SUM(monetary), 1) AS top20_revenue_share_pct
FROM ranked;
