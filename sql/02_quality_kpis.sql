-- Conditional aggregation; pass/fail outcome and overall failure rate.
SELECT COUNT(*) AS parts_total,
       SUM(CASE WHEN is_failed = 1 THEN 1 ELSE 0 END) AS failed_parts,
       SUM(CASE WHEN is_failed = 0 THEN 1 ELSE 0 END) AS passed_parts,
       ROUND(100.0 * AVG(is_failed), 2) AS failure_rate_pct
FROM parts;
