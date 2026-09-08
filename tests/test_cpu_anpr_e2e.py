from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.config import ROOT
from app.models import Alert, RecognitionAttempt, Sighting
from app.services.anpr import load_bgr
from app.services.cpu_anpr import cpu_anpr_status
from app.services.pipeline import process_frame_iter
from tests.conftest import add_camera, add_watchlist


def test_real_cpu_anpr_own_feed_persists_two_frames_before_alert(db):
    status = cpu_anpr_status()
    if not status.get("fast_alpr_available") or not status.get("models_ready"):
        pytest.skip("optional CPU ANPR models are not prepared on this host")
    frame_path = ROOT / "data" / "frames" / "cam-ahmedabad" / "0000.jpg"
    frame = load_bgr(Path(frame_path))
    if frame is None:
        pytest.skip("own-feed frame is unavailable")
    camera = add_camera(db, id="CPU-E2E", source_type="image_dir", target_analysis_fps=10.0)
    add_watchlist(db, "GJ01AB1234")
    result = process_frame_iter(
        db, camera,
        iter([(0, frame.copy(), 0.0), (1, frame.copy(), 100.0)]),
        run_id="cpu-e2e",
    )
    assert result["sightings"] == 2
    assert result["alerts"] == 1
    assert db.scalar(select(func.count(Alert.id))) == 1
    assert db.scalar(select(func.count(RecognitionAttempt.id))) == 2
    sightings = list(db.scalars(select(Sighting).order_by(Sighting.id)))
    assert [s.plate_norm for s in sightings] == ["GJ01AB1234", "GJ01AB1234"]
    assert sightings[0].vehicle_json["confirmation"]["status"] == "pending"
    assert sightings[1].vehicle_json["confirmation"]["status"] == "confirmed"
    alert = db.scalar(select(Alert))
    assert alert.sighting_id == sightings[1].id
