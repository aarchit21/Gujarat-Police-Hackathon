"""Bounded live government-feed probe. One catalogue camera only.

Never prints the access token. Does not open every stream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Alert, Camera, Sighting  # noqa: E402
from app.security import redact_url  # noqa: E402
from app.services.catalogue import sync_catalogue  # noqa: E402
from app.services.coverage import camera_origin, coverage  # noqa: E402
from app.services.network_check import host_network_report, probe_one_rtsp  # noqa: E402
from app.services.pipeline import analyze_camera  # noqa: E402
from app.services.reports import as_json, sighting_rows  # noqa: E402


def main() -> int:
    init_db()
    net = host_network_report(include_rtsp_probe=False)
    print("network", json.dumps(net, indent=2))
    db = SessionLocal()
    try:
        sync = sync_catalogue(db)
        print("catalogue_sync", json.dumps({k: sync[k] for k in sync if k != "payload"}, default=str))
        if not sync.get("ok"):
            print("BLOCKER: catalogue sync failed")
            return 2
        gov = [c for c in db.scalars(select(Camera)) if camera_origin(c) == "government_catalogue"]
        print("government_catalogue_count", len(gov))
        selected = next((c for c in gov if (c.source_uri or c.protected_rtsp_url_or_reference)), None)
        if selected is None:
            print("BLOCKER: catalogue cameras have no RTSP URL")
            return 3
        print("selected_camera", selected.id, "codec", selected.codec, "live", selected.catalogue_live)
        print("selected_rtsp", redact_url(selected.source_uri or selected.protected_rtsp_url_or_reference))
        probe = probe_one_rtsp(selected.source_uri or selected.protected_rtsp_url_or_reference)
        print("rtsp_probe", json.dumps(probe, indent=2))
        if not probe.get("frame"):
            print("BLOCKER: selected RTSP did not yield a frame")
            print("coverage", coverage(db))
            return 4
        before_s = {s.id for s in db.scalars(select(Sighting))}
        before_a = {a.id for a in db.scalars(select(Alert))}
        result = analyze_camera(db, selected.id, max_frames=12, max_seconds=15)
        print("bounded_anpr", json.dumps({k: result.get(k) for k in ("ok", "camera_id", "frames_seen", "frames_sampled", "sightings", "alerts", "analytics_active", "error")}, default=str))
        db.expire_all()
        new_s = [s for s in db.scalars(select(Sighting)) if s.id not in before_s and s.camera_id == selected.id]
        new_a = [a for a in db.scalars(select(Alert)) if a.id not in before_a]
        print("new_sightings", len(new_s), "new_alerts", len(new_a))
        if new_s:
            s = new_s[0]
            print("sample_sighting", s.plate_raw, s.plate_norm, "pts", s.source_pts_ms, "model", s.model_id)
        cov = coverage(db)
        print("coverage", json.dumps({k: cov[k] for k in ("onboarded_count", "own_feed_count", "government_catalogue_count", "connected_count", "analytics_active_count", "government_feed_status")}, default=str))
        print("report_rows", len(json.loads(as_json(sighting_rows(db)))))
        if not probe.get("frame"):
            return 4
        print("OK: catalogue reachable; selected stream decoded a frame. Watchlist hit only if a genuine plate matched.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
