from app.database import make_engine, parse_database_url
from app.services.cost import estimate


def test_sqlite_url_is_dev_fallback():
    info = parse_database_url("sqlite:///C:/tmp/cctv.db")
    assert info["is_sqlite"] is True
    assert info["is_postgresql"] is False


def test_postgres_url_redacts_password():
    info = parse_database_url("postgresql+psycopg://user:supersecret@db.example:5432/cctv")
    assert info["kind"] == "postgresql"
    assert "supersecret" not in info["safe_url"]
    assert info["host"] == "db.example"
    assert info["database"] == "cctv"


def test_sqlite_memory_engine():
    engine = make_engine("sqlite:///:memory:")
    assert engine.dialect.name == "sqlite"


def test_cost_estimate_labelled_and_uses_assumptions():
    out = estimate(
        {
            "camera_count": 100,
            "avg_bitrate_kbps": 2000,
            "target_analysis_fps": 2,
            "active_cameras": 4,
            "measured_worker_fps": 8,
            "gpu_hourly_cost": 1.5,
            "storage_cost_per_gb": 0.02,
            "evidence_events_per_day": 50,
            "avg_evidence_size_kb": 40,
        }
    )
    assert "Estimate only" in out["disclaimer"]
    assert out["estimated_source_bandwidth_mbps"] == 200.0
    assert out["approximate_workers_from_measured_throughput"] == 1
    assert out["approximate_monthly_compute_cost"] == 1 * 1.5 * 24 * 30
    assert out["estimated_evidence_storage_gb_per_day"] > 0
