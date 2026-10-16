-- NULL management, conditional aggregation and ordering by sensor data completeness.
SELECT sensor_id, COUNT(*) AS expected_readings,
       SUM(CASE WHEN reading IS NULL THEN 1 ELSE 0 END) AS missing_readings,
       ROUND(100.0 * SUM(CASE WHEN reading IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
           AS missing_pct
FROM sensor_readings
GROUP BY sensor_id
ORDER BY missing_pct DESC, sensor_id;
