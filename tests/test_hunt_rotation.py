"""Camera slot selection must rotate, not re-pin the same few cameras.

Regression: selection was `sorted by id` with PIN_DEFAULT forced to the front,
and strict decode tiering meant untested cameras never got a slot -- so they
stayed untested forever. On the real catalogue that left 25 of 32 cameras
unreachable while the same 6 were pinned on every click.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import settings
from app.models import Camera
from app.services.hunt import decode_ok_pin_ids, hunt_targets
from tests.conftest import add_camera


def _catalogue(db, *, ok: int = 6, untested: int = 20) -> None:
    for i in range(ok + untested):
        add_camera(
            db,
            id=f"cam{i:02d}",
            source_type="rtsp",
            source_uri=f"rtsp://10.0.0.1:8554/stream/cam{i:02d}",
            catalogue_live=True,
            decode_status="ok" if i < ok else "untested",
        )
    db.flush()


def test_untested_cameras_get_slots_instead_of_starving(db):
    _catalogue(db)
    picked = decode_ok_pin_ids(db, 4)
    statuses = {db.get(Camera, cid).decode_status for cid in picked}
    # Half the slots explore, so untested cameras are actually opened.
    assert "untested" in statuses
    assert "ok" in statuses


def test_repeated_clicks_reach_far_more_than_the_same_few(db):
    _catalogue(db)
    seen: set[str] = set()
    for _ in range(8):
        picked = decode_ok_pin_ids(db, 4)
        seen |= set(picked)
        for cid in picked:  # emulate mark_camera_hunted
            db.get(Camera, cid).last_hunted_at = datetime.now(timezone.utc)
        db.flush()
    # The old behaviour returned the same 4 ids every time.
    assert len(seen) > 10


def test_least_recently_hunted_is_preferred(db):
    _catalogue(db, ok=4, untested=0)
    now = datetime.now(timezone.utc)
    for offset, cid in enumerate(["cam00", "cam01", "cam02", "cam03"]):
        db.get(Camera, cid).last_hunted_at = now - timedelta(hours=offset)
    db.flush()
    # cam03 is the oldest, cam00 the most recent.
    picked = decode_ok_pin_ids(db, 2)
    assert "cam03" in picked
    assert "cam00" not in picked


def test_never_hunted_cameras_come_first(db):
    _catalogue(db, ok=4, untested=0)
    now = datetime.now(timezone.utc)
    for cid in ["cam00", "cam01", "cam02"]:
        db.get(Camera, cid).last_hunted_at = now
    db.flush()  # cam03 has never been hunted
    assert "cam03" in decode_ok_pin_ids(db, 1)


def test_decode_failed_cameras_are_last(db):
    _catalogue(db, ok=1, untested=1)
    add_camera(db, id="cam-dead", source_type="rtsp",
               source_uri="rtsp://10.0.0.1:8554/stream/dead",
               catalogue_live=True, decode_status="failed")
    db.flush()
    assert "cam-dead" not in decode_ok_pin_ids(db, 2)


def test_explore_fraction_zero_restores_decode_ok_only(db, monkeypatch):
    monkeypatch.setattr(settings, "hunt_explore_fraction", 0.0)
    _catalogue(db)
    picked = decode_ok_pin_ids(db, 4)
    assert all(db.get(Camera, cid).decode_status == "ok" for cid in picked)


def test_fixed_mode_reproduces_the_old_deterministic_order(db, monkeypatch):
    monkeypatch.setattr(settings, "hunt_rotation", "fixed")
    _catalogue(db)
    assert decode_ok_pin_ids(db, 4) == decode_ok_pin_ids(db, 4)


def test_seed_makes_selection_reproducible(db, monkeypatch):
    monkeypatch.setattr(settings, "hunt_pin_seed", 1234)
    _catalogue(db)
    assert decode_ok_pin_ids(db, 4) == decode_ok_pin_ids(db, 4)


def test_explicit_pin_ids_are_honoured_in_order(db):
    _catalogue(db)
    targets = hunt_targets(db, pinned_only=True, pin_ids=["cam07", "cam03"])
    assert [c.id for c in targets] == ["cam07", "cam03"]


def test_slots_are_filled_even_when_a_pool_is_short(db):
    _catalogue(db, ok=1, untested=1)
    assert len(decode_ok_pin_ids(db, 4)) == 2
