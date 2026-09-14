"""Safe schema initialisation. Adds columns/indexes; never drops tables or user rows."""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


CAMERA_COLUMNS = {
    "substream_uri": ("TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    "priority_class": ("VARCHAR(8) DEFAULT 'D'", "VARCHAR(8) DEFAULT 'D'"),
    "processing_mode": ("VARCHAR(32) DEFAULT 'deferred'", "VARCHAR(32) DEFAULT 'deferred'"),
    "analytics_policy": ("VARCHAR(32) DEFAULT 'on_demand'", "VARCHAR(32) DEFAULT 'on_demand'"),
    "compute_target": ("VARCHAR(64) DEFAULT ''", "VARCHAR(64) DEFAULT ''"),
    "network_class": ("VARCHAR(32) DEFAULT 'offline'", "VARCHAR(32) DEFAULT 'offline'"),
    "target_analysis_fps": ("DOUBLE PRECISION", "REAL"),
    "capabilities": ("TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    "vendor": ("VARCHAR(80) DEFAULT ''", "VARCHAR(80) DEFAULT ''"),
    "model": ("VARCHAR(80) DEFAULT ''", "VARCHAR(80) DEFAULT ''"),
    "clock_offset_ms": ("DOUBLE PRECISION", "REAL"),
    "catalogue_camera_id": ("VARCHAR(128)", "VARCHAR(128)"),
    "catalogue_live": ("BOOLEAN DEFAULT FALSE", "INTEGER DEFAULT 0"),
    "codec": ("VARCHAR(32) DEFAULT ''", "VARCHAR(32) DEFAULT ''"),
    "width": ("INTEGER", "INTEGER"),
    "height": ("INTEGER", "INTEGER"),
    "reported_fps": ("DOUBLE PRECISION", "REAL"),
    "bitrate": ("INTEGER", "INTEGER"),
    "protected_rtsp_url_or_reference": ("TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    "whep_url": ("TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    "hls_url": ("TEXT DEFAULT ''", "TEXT DEFAULT ''"),
    "catalogue_synced_at": ("TIMESTAMPTZ", "DATETIME"),
    "decode_tested_at": ("TIMESTAMPTZ", "DATETIME"),
    "decode_status": ("VARCHAR(32) DEFAULT 'untested'", "VARCHAR(32) DEFAULT 'untested'"),
    "source_pts_ms": ("DOUBLE PRECISION", "REAL"),
    "last_pts_ms": ("DOUBLE PRECISION", "REAL"),
    "reconnect_count": ("INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
    "active_protocol": ("VARCHAR(16) DEFAULT ''", "VARCHAR(16) DEFAULT ''"),
    "measured_worker_fps": ("DOUBLE PRECISION", "REAL"),
    "measured_at": ("TIMESTAMPTZ", "DATETIME"),
    "coords_source": ("VARCHAR(24) DEFAULT ''", "VARCHAR(24) DEFAULT ''"),
    "last_hunted_at": ("TIMESTAMPTZ", "DATETIME"),
    "plate_recognition_mode": ("VARCHAR(12) DEFAULT 'inherit'", "VARCHAR(12) DEFAULT 'inherit'"),
}

SIGHTING_COLUMNS = {
    "source_pts_ms": ("DOUBLE PRECISION", "REAL"),
    "provider": ("VARCHAR(32) DEFAULT 'local'", "VARCHAR(32) DEFAULT 'local'"),
    "vendor_event_id": ("VARCHAR(128)", "VARCHAR(128)"),
    "vendor_payload_hash": ("VARCHAR(64) DEFAULT ''", "VARCHAR(64) DEFAULT ''"),
    "bbox_x": ("INTEGER", "INTEGER"),
    "bbox_y": ("INTEGER", "INTEGER"),
    "bbox_w": ("INTEGER", "INTEGER"),
    "bbox_h": ("INTEGER", "INTEGER"),
    "frame_width": ("INTEGER", "INTEGER"),
    "frame_height": ("INTEGER", "INTEGER"),
    "vehicle_type": ("VARCHAR(32) DEFAULT ''", "VARCHAR(32) DEFAULT ''"),
    "vehicle_make": ("VARCHAR(40) DEFAULT ''", "VARCHAR(40) DEFAULT ''"),
    "vehicle_model": ("VARCHAR(40) DEFAULT ''", "VARCHAR(40) DEFAULT ''"),
    "vehicle_color": ("VARCHAR(40) DEFAULT ''", "VARCHAR(40) DEFAULT ''"),
    "vehicle_json": ("JSONB", "TEXT DEFAULT ''"),
    "vehicle_observation_id": ("INTEGER", "INTEGER"),
}

# Human verification of vehicle attributes. Additive only; the raw model
# output stays in metadata_json and is never dropped.
VEHICLE_OBSERVATION_COLUMNS = {
    "verified_vehicle_type": ("VARCHAR(32) DEFAULT ''", "VARCHAR(32) DEFAULT ''"),
    "verified_vehicle_color": ("VARCHAR(32) DEFAULT ''", "VARCHAR(32) DEFAULT ''"),
    "verified_by": ("VARCHAR(64) DEFAULT ''", "VARCHAR(64) DEFAULT ''"),
    "verified_at": ("TIMESTAMPTZ", "DATETIME"),
    "review_status": ("VARCHAR(24) DEFAULT 'unreviewed'", "VARCHAR(24) DEFAULT 'unreviewed'"),
}

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS ix_vehicle_observations_review_status ON vehicle_observations (review_status)",
    "CREATE INDEX IF NOT EXISTS ix_vehicle_observations_verified_type ON vehicle_observations (verified_vehicle_type)",
    "CREATE INDEX IF NOT EXISTS ix_cameras_catalogue_camera_id ON cameras (catalogue_camera_id)",
    "CREATE INDEX IF NOT EXISTS ix_sightings_camera_id ON sightings (camera_id)",
    "CREATE INDEX IF NOT EXISTS ix_sightings_plate_norm ON sightings (plate_norm)",
    "CREATE INDEX IF NOT EXISTS ix_sightings_source_time ON sightings (source_time)",
    "CREATE INDEX IF NOT EXISTS ix_watchlist_plate_norm ON watchlist (plate_norm)",
    "CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts (status)",
    "CREATE INDEX IF NOT EXISTS ix_alerts_plate_norm ON alerts (plate_norm)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_alert_dedup ON alerts (watchlist_id, camera_id, passage_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_sightings_vendor_event ON sightings (vendor_event_id)",
    "CREATE INDEX IF NOT EXISTS ix_sightings_vehicle_observation_id ON sightings (vehicle_observation_id)",
    "CREATE INDEX IF NOT EXISTS ix_vehicle_observation_camera_time ON vehicle_observations (camera_id, first_seen_at)",
    "CREATE INDEX IF NOT EXISTS ix_vehicle_observation_type_color_time ON vehicle_observations (vehicle_type, vehicle_color, first_seen_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_vehicle_observation_track ON vehicle_observations (camera_id, run_id, track_id)",
]


def _dialect(engine: Engine) -> str:
    name = engine.dialect.name
    if name.startswith("postgres"):
        return "postgresql"
    return name


def _columns(engine: Engine, table: str) -> set[str]:
    insp = inspect(engine)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _add_columns(engine: Engine, table: str, spec: dict[str, tuple[str, str]]) -> list[str]:
    added: list[str] = []
    existing = _columns(engine, table)
    if not existing:
        return added
    pg = _dialect(engine) == "postgresql"
    with engine.begin() as conn:
        for name, (pg_type, sqlite_type) in spec.items():
            if name in existing:
                continue
            ddl = pg_type if pg else sqlite_type
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
            added.append(f"{table}.{name}")
    return added


def _try_postgis(engine: Engine) -> bool:
    if _dialect(engine) != "postgresql":
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            cols = {row[0] for row in conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name='cameras'"
            ))}
            if "geom" not in cols:
                conn.execute(text("ALTER TABLE cameras ADD COLUMN geom geography(Point, 4326)"))
            conn.execute(text(
                """
                UPDATE cameras
                SET geom = ST_SetSRID(ST_MakePoint(lng, lat), 4326)::geography
                WHERE lat IS NOT NULL AND lng IS NOT NULL
                  AND (geom IS NULL OR geom IS NOT NULL)
                """
            ))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cameras_geom ON cameras USING GIST (geom)"))
        return True
    except Exception:
        return False


def _postgres_vehicle_jsonb(engine: Engine) -> bool:
    if _dialect(engine) != "postgresql":
        return False
    try:
        with engine.begin() as conn:
            col = conn.execute(
                text(
                    """
                    SELECT data_type FROM information_schema.columns
                    WHERE table_name='sightings' AND column_name='vehicle_json'
                    """
                )
            ).scalar()
            if col and col.lower() in {"text", "character varying"}:
                conn.execute(
                    text(
                        """
                        ALTER TABLE sightings
                        ALTER COLUMN vehicle_json TYPE jsonb
                        USING CASE
                          WHEN vehicle_json IS NULL OR vehicle_json = '' THEN NULL
                          ELSE vehicle_json::jsonb
                        END
                        """
                    )
                )
        return True
    except Exception:
        return False


def _sqlite_empty_vehicle_json(engine: Engine) -> None:
    if _dialect(engine) != "sqlite":
        return


def _backfill_vehicle_observations(engine: Engine) -> None:
    """Create additive investigation records from existing attributed sightings."""
    if "vehicle_observations" not in inspect(engine).get_table_names():
        return
    dialect = _dialect(engine)
    insert_prefix = "INSERT OR IGNORE" if dialect == "sqlite" else "INSERT"
    conflict = "" if dialect == "sqlite" else " ON CONFLICT (camera_id, run_id, track_id) DO NOTHING"
    try:
        with engine.begin() as conn:
            conn.execute(text(
                f"""
                {insert_prefix} INTO vehicle_observations
                (camera_id, run_id, track_id, first_seen_at, last_seen_at, first_pts_ms, last_pts_ms,
                 best_frame_index, bbox_x, bbox_y, bbox_w, bbox_h, frame_width, frame_height,
                 detector, detector_confidence, vehicle_type, type_confidence, type_source,
                 vehicle_color, color_confidence, color_source, evidence_path, context_evidence_path, metadata_json, created_at, updated_at)
                SELECT camera_id, COALESCE(run_id, ''), COALESCE(passage_id, 'legacy-' || id),
                       source_time, source_time, source_pts_ms, source_pts_ms, frame_index,
                       bbox_x, bbox_y, bbox_w, bbox_h, frame_width, frame_height,
                       COALESCE(model_id, ''), COALESCE(confidence, 0),
                       CASE WHEN COALESCE(vehicle_type, '') = '' THEN 'unknown' ELSE vehicle_type END,
                       COALESCE(confidence, 0), 'legacy_sighting',
                       CASE WHEN COALESCE(vehicle_color, '') = '' THEN 'unknown' ELSE vehicle_color END,
                       0, 'legacy_sighting', COALESCE(evidence_path, ''), '', NULL, ingest_time, ingest_time
                FROM sightings
                WHERE COALESCE(vehicle_type, '') <> '' OR COALESCE(vehicle_color, '') <> ''
                {conflict}
                """
            ))
            conn.execute(text(
                """
                UPDATE sightings
                SET vehicle_observation_id = (
                    SELECT vo.id FROM vehicle_observations vo
                    WHERE vo.camera_id = sightings.camera_id
                      AND vo.run_id = COALESCE(sightings.run_id, '')
                      AND vo.track_id = COALESCE(sightings.passage_id, 'legacy-' || sightings.id)
                    LIMIT 1
                )
                WHERE vehicle_observation_id IS NULL
                  AND (COALESCE(vehicle_type, '') <> '' OR COALESCE(vehicle_color, '') <> '')
                """
            ))
    except Exception:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE sightings SET vehicle_json = NULL WHERE vehicle_json IS NULL OR trim(vehicle_json) = ''"))
    except Exception:
        return


def apply_migrations(engine: Engine) -> dict:
    added = []
    added.extend(_add_columns(engine, "cameras", CAMERA_COLUMNS))
    added.extend(_add_columns(engine, "sightings", SIGHTING_COLUMNS))
    added.extend(_add_columns(engine, "vehicle_observations", VEHICLE_OBSERVATION_COLUMNS))
    _postgres_vehicle_jsonb(engine)
    _sqlite_empty_vehicle_json(engine)
    _backfill_vehicle_observations(engine)
    with engine.begin() as conn:
        for stmt in INDEX_SQL:
            try:
                conn.execute(text(stmt))
            except Exception:
                continue
    postgis = _try_postgis(engine)
    return {
        "added_columns": added,
        "postgis": postgis,
        "dialect": _dialect(engine),
        "destroyed_data": False,
    }
