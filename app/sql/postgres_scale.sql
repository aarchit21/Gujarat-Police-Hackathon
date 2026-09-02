-- PostgreSQL / PostGIS scale notes for a future statewide deployment.
-- These statements are NOT applied automatically. P0 does not claim an 80,000-camera load test.

-- Enable PostGIS (also attempted at init when DATABASE_URL is PostgreSQL).
CREATE EXTENSION IF NOT EXISTS postgis;

-- Camera location as geography for spatial queries.
-- ALTER TABLE cameras ADD COLUMN IF NOT EXISTS geom geography(Point, 4326);
-- UPDATE cameras SET geom = ST_SetSRID(ST_MakePoint(lng, lat), 4326)::geography
--   WHERE lat IS NOT NULL AND lng IS NOT NULL;
-- CREATE INDEX IF NOT EXISTS ix_cameras_geom ON cameras USING GIST (geom);

-- Example nearest-camera query:
-- SELECT id, name, ST_Distance(geom, ST_SetSRID(ST_MakePoint(72.57, 23.02), 4326)::geography) AS metres
-- FROM cameras
-- ORDER BY geom <-> ST_SetSRID(ST_MakePoint(72.57, 23.02), 4326)::geography
-- LIMIT 20;

-- Future time-based partitioning of sightings (requires converting the table):
-- CREATE TABLE sightings_partitioned (LIKE sightings INCLUDING ALL)
--   PARTITION BY RANGE (source_time);
-- CREATE TABLE sightings_y2026m09 PARTITION OF sightings_partitioned
--   FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

-- Connection pooling: set DB_POOL_SIZE and DB_MAX_OVERFLOW in the environment.
-- Production recommendations (not implemented in P0):
--   * streaming replication (primary + hot standby)
--   * pg_dump / WAL archiving backups
--   * retention job that archives evidence metadata after the departmental policy window
-- Departmental full video remains local and is never copied here.
