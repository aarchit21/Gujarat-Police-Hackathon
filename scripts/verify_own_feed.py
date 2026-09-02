"""Run own-feed analysis and print honest persistence checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Alert, Camera, Sighting  # noqa: E402
from app.services.coverage import coverage  # noqa: E402
from app.services.pipeline import analyze_camera  # noqa: E402
from scripts.seed import main as seed_main  # noqa: E402


def main() -> int:
    seed_main([])
    init_db()
    db = SessionLocal()
    try:
        own = list(db.scalars(select(Camera).where(Camera.source_type == "image_dir")))
        results = [analyze_camera(db, c.id) for c in own]
        sightings = list(db.scalars(select(Sighting)))
        alerts = list(db.scalars(select(Alert)))
        evidence = [s for s in sightings if s.evidence_path]
        cov = coverage(db)
        print("cameras_processed", len(results), results)
        print("sightings", len(sightings))
        print("alerts", len(alerts))
        print("evidence_rows", len(evidence))
        print("coverage", cov["honest_coverage"], "gov", cov["government_feed_status"])
        if not sightings:
            print("FAIL: no Sighting rows persisted")
            return 1
        if not alerts:
            print("FAIL: no Alert rows (exact watchlist match did not fire)")
            return 1
        if not evidence:
            print("FAIL: no evidence crops")
            return 1
        print("OK own-feed: sighting, alert, evidence persisted")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
