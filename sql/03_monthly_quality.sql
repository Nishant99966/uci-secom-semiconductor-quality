-- GROUP BY, CTE and window function; prior-month failure-rate comparison.
WITH monthly AS (
  SELECT SUBSTR(recorded_at, 1, 7) AS production_month,
         COUNT(*) AS parts_total,
         SUM(is_failed) AS failed_parts,
         ROUND(100.0 * AVG(is_failed), 2) AS failure_rate_pct
  FROM parts
  GROUP BY SUBSTR(recorded_at, 1, 7)
)
SELECT production_month, parts_total, failed_parts, failure_rate_pct,
       LAG(failure_rate_pct) OVER (ORDER BY production_month) AS prior_month_failure_rate_pct,
       ROUND(failure_rate_pct - LAG(failure_rate_pct) OVER (ORDER BY production_month), 2)
         AS change_in_percentage_points
FROM monthly
ORDER BY production_month;
