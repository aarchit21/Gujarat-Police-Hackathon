from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    department: Mapped[str] = mapped_column(String(80))
    city: Mapped[str] = mapped_column(String(80), default="")
    lat: Mapped[float] = mapped_column(Float, default=0.0)
    lng: Mapped[float] = mapped_column(Float, default=0.0)
    source_type: Mapped[str] = mapped_column(String(32), default="blocked")
    source_uri: Mapped[str] = mapped_column(Text, default="")
    substream_uri: Mapped[str] = mapped_column(Text, default="")
    priority_class: Mapped[str] = mapped_column(String(8), default="D")
    processing_mode: Mapped[str] = mapped_column(String(32), default="deferred")
    analytics_policy: Mapped[str] = mapped_column(String(32), default="on_demand")
    compute_target: Mapped[str] = mapped_column(String(64), default="")
    network_class: Mapped[str] = mapped_column(String(32), default="offline")
    target_analysis_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="deferred")
    status_reason: Mapped[str] = mapped_column(Text, default="")
    analytics_active: Mapped[bool] = mapped_column(Boolean, default=False)
    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    capabilities: Mapped[str] = mapped_column(Text, default="")
    vendor: Mapped[str] = mapped_column(String(80), default="")
    model: Mapped[str] = mapped_column(String(80), default="")
    clock_offset_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    catalogue_camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    catalogue_live: Mapped[bool] = mapped_column(Boolean, default=False)
    codec: Mapped[str] = mapped_column(String(32), default="")
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reported_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    bitrate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    protected_rtsp_url_or_reference: Mapped[str] = mapped_column(Text, default="")
    whep_url: Mapped[str] = mapped_column(Text, default="")
    hls_url: Mapped[str] = mapped_column(Text, default="")
    catalogue_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decode_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decode_status: Mapped[str] = mapped_column(String(32), default="untested")
    source_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    reconnect_count: Mapped[int] = mapped_column(Integer, default=0)
    active_protocol: Mapped[str] = mapped_column(String(16), default="")

    sightings: Mapped[list["Sighting"]] = relationship(back_populates="camera")

    @property
    def latitude(self) -> float:
        return self.lat

    @property
    def longitude(self) -> float:
        return self.lng


class WatchlistEntry(Base):
    __tablename__ = "watchlist"
    __table_args__ = (Index("ix_watchlist_active_plate", "active", "plate_norm"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_raw: Mapped[str] = mapped_column(String(32))
    plate_norm: Mapped[str] = mapped_column(String(32), index=True)
    purpose: Mapped[str] = mapped_column(String(80), default="stolen_vehicle")
    priority: Mapped[str] = mapped_column(String(16), default="high")
    authority: Mapped[str] = mapped_column(String(80), default="demo-synthetic")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str] = mapped_column(Text, default="Synthetic representative record. Not a real GJ plate.")


class Sighting(Base):
    __tablename__ = "sightings"
    __table_args__ = (
        Index("ix_sightings_camera_time", "camera_id", "source_time"),
        UniqueConstraint("vendor_event_id", name="uq_sightings_vendor_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    passage_id: Mapped[str] = mapped_column(String(64), index=True)
    source_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ingest_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_raw: Mapped[str] = mapped_column(String(64))
    plate_norm: Mapped[str] = mapped_column(String(32), index=True)
    plate_voted: Mapped[str] = mapped_column(String(32), default="")
    syntax_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model_id: Mapped[str] = mapped_column(String(80), default="tesseract-opencv-p0")
    model_hash: Mapped[str] = mapped_column(String(64), default="unpinned")
    evidence_path: Mapped[str] = mapped_column(Text, default="")
    run_id: Mapped[str] = mapped_column(String(64), default="")
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    provider: Mapped[str] = mapped_column(String(32), default="local")
    vendor_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    vendor_payload_hash: Mapped[str] = mapped_column(String(64), default="")
    bbox_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="sightings")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("watchlist_id", "camera_id", "passage_id", name="uq_alert_dedup"),
        Index("ix_alerts_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"))
    watchlist_id: Mapped[int] = mapped_column(ForeignKey("watchlist.id"))
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    passage_id: Mapped[str] = mapped_column(String(64))
    plate_norm: Mapped[str] = mapped_column(String(32), index=True)
    match_type: Mapped[str] = mapped_column(String(16), default="exact")
    status: Mapped[str] = mapped_column(String(24), default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sighting: Mapped[Sighting] = relationship()
    watchlist: Mapped[WatchlistEntry] = relationship()


class AuditEvent(Base):
    __tablename__ = "audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(80), default="operator")
    action: Mapped[str] = mapped_column(String(80), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")


class SystemState(Base):
    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
