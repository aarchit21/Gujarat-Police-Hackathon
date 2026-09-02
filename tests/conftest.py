from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.database import init_db, make_engine, make_session_factory
from app.models import Camera, WatchlistEntry
from app.services.anpr import PlateRead
from app.services.plates import normalize


@pytest.fixture
def db() -> Session:
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session = make_session_factory(engine)()
    try:
        yield session
    finally:
        session.close()


def add_camera(db: Session, **kwargs) -> Camera:
    defaults = dict(
        id="CAM-TEST-001",
        name="test",
        department="Home",
        city="Ahmedabad",
        lat=23.02,
        lng=72.57,
        source_type="image_dir",
        source_uri="",
        status="onboarded",
        processing_mode="local_worker",
        analytics_policy="continuous",
        priority_class="A",
        network_class="good",
        analytics_active=False,
    )
    defaults.update(kwargs)
    cam = Camera(**defaults)
    db.add(cam)
    db.commit()
    return cam


def add_watchlist(db: Session, plate: str = "GJ01AB1234") -> WatchlistEntry:
    row = WatchlistEntry(plate_raw=plate, plate_norm=normalize(plate), purpose="stolen_vehicle")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def fake_read(_bgr) -> PlateRead:
    crop = np.zeros((20, 60, 3), dtype=np.uint8)
    return PlateRead("GJ01AB1234", "GJ01AB1234", True, 0.9, crop, (2, 2, 60, 20))


class FakeCap:
    def __init__(self, opened=True, frames=None, pts=None):
        self._opened = opened
        self.frames = frames or []
        self.pts = pts or []
        self.i = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        frame = self.frames[self.i]
        self.i += 1
        return True, frame

    def get(self, prop):
        idx = max(0, self.i - 1)
        if prop == 0:
            return self.pts[idx] if idx < len(self.pts) else 0.0
        if prop == 3:
            return 1280
        if prop == 4:
            return 720
        if prop == 5:
            raise AssertionError("CAP_PROP_FPS must not be used for timing")
        return 0

    def release(self):
        self.released = True
        self._opened = False
