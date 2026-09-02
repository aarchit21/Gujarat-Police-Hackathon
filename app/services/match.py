from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Alert, Sighting, WatchlistEntry
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


def rematch_watchlist_entry(db: Session, watchlist: WatchlistEntry) -> dict:
    """Create exact alerts from already-persisted sightings. Never invents plates."""
    from app.services.serialize import plate_keys

    if not watchlist.active:
        return {"scanned": 0, "alerts_created": 0, "plate_norm": watchlist.plate_norm}
    created = 0
    scanned = 0
    sightings = list(db.scalars(select(Sighting).order_by(Sighting.id)))
    target = normalize(watchlist.plate_norm)
    for sighting in sightings:
        scanned += 1
        if target not in plate_keys(sighting):
            continue
        _alert, new = open_alert(
            db,
            sighting_id=sighting.id,
            watchlist=watchlist,
            camera_id=sighting.camera_id,
            passage_id=sighting.passage_id,
            plate_norm=watchlist.plate_norm,
        )
        if new:
            created += 1
    return {"scanned": scanned, "alerts_created": created, "plate_norm": watchlist.plate_norm}


def observed_plates(db: Session) -> list[dict]:
    active = {w.plate_norm for w in db.scalars(select(WatchlistEntry).where(WatchlistEntry.active.is_(True)))}
    rows: dict[str, dict] = {}
    for s in db.scalars(select(Sighting).order_by(Sighting.source_time)):
        key = s.plate_norm or s.plate_voted
        if not key:
            continue
        rec = rows.get(key)
        if rec is None:
            rec = {
                "plate_norm": key,
                "plate_raw": s.plate_raw,
                "count": 0,
                "cameras": set(),
                "last_time": None,
                "last_camera": "",
                "watchlisted": key in active,
                "model_id": s.model_id,
                "syntax_ok": bool(s.syntax_ok),
                "provider": s.provider,
            }
            rows[key] = rec
        rec["count"] += 1
        rec["cameras"].add(s.camera_id)
        rec["last_time"] = s.source_time.isoformat() if s.source_time else rec["last_time"]
        rec["last_camera"] = s.camera_id
        rec["watchlisted"] = key in active
    out = []
    for rec in rows.values():
        item = dict(rec)
        item["cameras"] = sorted(item["cameras"])
        out.append(item)
    out.sort(key=lambda r: (-int(r["count"]), r["plate_norm"]))
    return out
