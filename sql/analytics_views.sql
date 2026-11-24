-- 1. Churn & Revenue Lost by Tenure Bucket (CTE + Window Functions)
CREATE OR REPLACE VIEW v_churn_by_tenure AS
WITH bucketed AS (
    SELECT
        customer_id,
        monthly_charges,
        churn,
        CASE
            WHEN tenure <= 12 THEN '0-12 Months'
            WHEN tenure <= 24 THEN '13-24 Months'
            WHEN tenure <= 48 THEN '25-48 Months'
            ELSE '49+ Months'
        END AS tenure_bucket
    FROM telco_customers
),
agg AS (
    SELECT
        tenure_bucket,
        COUNT(*) AS total_customers,
        SUM(churn) AS churned_customers,
        ROUND(AVG(churn) * 100, 2) AS churn_rate_pct,
        ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
    FROM bucketed
    GROUP BY tenure_bucket
)
SELECT
    tenure_bucket,
    total_customers,
    churned_customers,
    churn_rate_pct,
    monthly_revenue_lost,
    ROUND(100.0 * churned_customers / SUM(churned_customers) OVER (), 2) AS pct_of_all_churners,
    RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank
FROM agg
ORDER BY risk_rank;

-- 2. Churn by Payment Method vs. Company Average (CTE + Window Functions)
CREATE OR REPLACE VIEW v_churn_by_payment AS
WITH agg AS (
    SELECT
        payment_method,
        COUNT(*) AS total_customers,
        SUM(churn) AS churned_customers,
        ROUND(AVG(churn) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge
    FROM telco_customers
    GROUP BY payment_method
)
SELECT
    payment_method,
    total_customers,
    churned_customers,
    churn_rate_pct,
    ROUND(
        churn_rate_pct - (100.0 * SUM(churned_customers) OVER () / SUM(total_customers) OVER ()),
        2
    ) AS diff_vs_company_avg_pct,
    avg_monthly_charge,
    RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank
FROM agg
ORDER BY risk_rank;

-- 3. Multi-Dimensional Segment Risk View (Contract x Tenure x Internet Service)
CREATE OR REPLACE VIEW v_churn_by_segment AS
WITH segmented AS (
    SELECT
        contract,
        internet_service,
        monthly_charges,
        churn,
        CASE
            WHEN tenure <= 12 THEN '0-12 Months'
            WHEN tenure <= 24 THEN '13-24 Months'
            WHEN tenure <= 48 THEN '25-48 Months'
            ELSE '49+ Months'
        END AS tenure_bucket
    FROM telco_customers
),
agg AS (
    SELECT
        contract,
        tenure_bucket,
        internet_service,
        COUNT(*) AS total_customers,
        SUM(churn) AS churned_customers,
        ROUND(AVG(churn) * 100, 2) AS churn_rate_pct,
        ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
    FROM segmented
    GROUP BY contract, tenure_bucket, internet_service
)
SELECT
    contract,
    tenure_bucket,
    internet_service,
    total_customers,
    churned_customers,
    churn_rate_pct,
    monthly_revenue_lost,
    ROUND(100.0 * monthly_revenue_lost / SUM(monthly_revenue_lost) OVER (), 2) AS pct_of_revenue_lost,
    DENSE_RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank
FROM agg
ORDER BY risk_rank, monthly_revenue_lost DESC;