from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Alert, WatchlistEntry
from app.services.plates import normalize


def exact_watchlist(db: Session, plate_norm: str) -> WatchlistEntry | None:
    key = normalize(plate_norm)
    if not key:
        return None
    return db.scalar(
        select(WatchlistEntry).where(
            WatchlistEntry.active.is_(True),
            WatchlistEntry.plate_norm == key,
        )
    )


def open_alert(
    db: Session,
    *,
    sighting_id: int,
    watchlist: WatchlistEntry,
    camera_id: str,
    passage_id: str,
    plate_norm: str,
) -> tuple[Alert, bool]:
    if not sighting_id:
        raise ValueError("alert requires a persisted sighting id")
    existing = db.scalar(
        select(Alert).where(
            Alert.watchlist_id == watchlist.id,
            Alert.camera_id == camera_id,
            Alert.passage_id == passage_id,
        )
    )
    if existing:
        return existing, False
    alert = Alert(
        sighting_id=sighting_id,
        watchlist_id=watchlist.id,
        camera_id=camera_id,
        passage_id=passage_id,
        plate_norm=plate_norm,
        match_type="exact",
        status="new",
    )
    db.add(alert)
    db.flush()
    return alert, True


def match_sighting(db: Session, sighting) -> tuple[object | None, bool]:
    from app.services.plates import layout_hint

    if not getattr(sighting, "id", None):
        raise ValueError("match_sighting requires a persisted Sighting row")
    keys = {
        sighting.plate_norm,
        sighting.plate_voted,
        layout_hint(sighting.plate_norm or ""),
        layout_hint(sighting.plate_voted or ""),
    }
    for key in keys:
        hit = exact_watchlist(db, key)
        if hit:
            return open_alert(
                db,
                sighting_id=sighting.id,
                watchlist=hit,
                camera_id=sighting.camera_id,
                passage_id=sighting.passage_id,
                plate_norm=hit.plate_norm,
            )
    return None, False
