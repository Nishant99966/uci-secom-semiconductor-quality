PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS parts (
  part_id INTEGER PRIMARY KEY,
  recorded_at TEXT NOT NULL,
  is_failed INTEGER NOT NULL CHECK (is_failed IN (0, 1))
);

CREATE TABLE IF NOT EXISTS sensor_readings (
  part_id INTEGER NOT NULL,
  sensor_id INTEGER NOT NULL CHECK (sensor_id BETWEEN 1 AND 590),
  reading REAL,
  PRIMARY KEY (part_id, sensor_id),
  FOREIGN KEY (part_id) REFERENCES parts(part_id)
);

CREATE INDEX IF NOT EXISTS idx_parts_time ON parts(recorded_at);
CREATE INDEX IF NOT EXISTS idx_parts_failed ON parts(is_failed);
CREATE INDEX IF NOT EXISTS idx_readings_sensor ON sensor_readings(sensor_id);
