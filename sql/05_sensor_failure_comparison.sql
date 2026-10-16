-- JOIN + conditional AVG; exploratory comparison, not proof of defect causality.
WITH outcomes AS (
  SELECT r.sensor_id,
         COUNT(r.reading) AS observed_readings,
         SUM(CASE WHEN p.is_failed = 1 AND r.reading IS NOT NULL THEN 1 ELSE 0 END)
           AS observed_on_failed_parts,
         SUM(CASE WHEN p.is_failed = 0 AND r.reading IS NOT NULL THEN 1 ELSE 0 END)
           AS observed_on_passed_parts,
         AVG(CASE WHEN p.is_failed = 1 THEN r.reading END) AS failed_mean,
         AVG(CASE WHEN p.is_failed = 0 THEN r.reading END) AS passed_mean
  FROM sensor_readings AS r
  JOIN parts AS p ON p.part_id = r.part_id
  GROUP BY r.sensor_id
)
SELECT sensor_id, observed_readings, observed_on_failed_parts, observed_on_passed_parts,
       ROUND(failed_mean, 6) AS failed_mean,
       ROUND(passed_mean, 6) AS passed_mean,
       ROUND(failed_mean - passed_mean, 6) AS difference_failed_minus_passed
FROM outcomes
WHERE observed_on_failed_parts >= 10 AND observed_on_passed_parts >= 10
ORDER BY ABS(failed_mean - passed_mean) DESC, sensor_id
LIMIT 25;
