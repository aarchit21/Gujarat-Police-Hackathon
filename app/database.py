from __future__ import annotations

import json
from collections.abc import Generator
from urllib.parse import urlparse

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings


def soft_json_loads(value):
    """SQLite older rows used '' for vehicle_json; JSON() cannot json.loads that."""
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "null":
            return None
        return json.loads(text)
    return value


class Base(DeclarativeBase):
    pass


def parse_database_url(url: str) -> dict:
    """Return dialect info without leaking credentials."""
    raw = (url or "").strip()
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme.startswith("postgres"):
        kind = "postgresql"
    elif scheme.startswith("sqlite"):
        kind = "sqlite"
    else:
        kind = scheme or "unknown"
    host = parsed.hostname or ""
    port = parsed.port
    dbname = (parsed.path or "").lstrip("/")
    if kind == "sqlite":
        safe = f"sqlite:///{dbname or raw.removeprefix('sqlite:///')}"
    else:
        auth = "***:***@" if parsed.username or parsed.password else ""
        netloc = host
        if port:
            netloc = f"{host}:{port}"
        safe = f"{scheme}://{auth}{netloc}/{dbname}"
    return {
        "kind": kind,
        "dialect": kind,
        "driver": scheme,
        "host": host,
        "port": port,
        "database": dbname,
        "safe_url": safe,
        "is_sqlite": kind == "sqlite",
        "is_postgresql": kind == "postgresql",
    }


def make_engine(url: str | None = None, *, pool_size: int | None = None, max_overflow: int | None = None) -> Engine:
    target = url or settings.database_url
    info = parse_database_url(target)
    if info["is_sqlite"]:
        kwargs: dict = {
            "connect_args": {"check_same_thread": False},
            "future": True,
            "json_serializer": json.dumps,
            "json_deserializer": soft_json_loads,
        }
        if ":memory:" in target:
            kwargs["poolclass"] = StaticPool
        engine = create_engine(target, **kwargs)

        @event.listens_for(engine, "connect")
        def _sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(
        target,
        future=True,
        json_serializer=json.dumps,
        json_deserializer=soft_json_loads,
        pool_size=pool_size if pool_size is not None else settings.db_pool_size,
        max_overflow=max_overflow if max_overflow is not None else settings.db_max_overflow,
        pool_pre_ping=True,
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)
_postgis_enabled = False


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(bind: Engine | None = None) -> dict:
    """Create tables, add missing columns, indexes. Never drops user data."""
    from app import models as _models  # noqa: F401
    from app.migrate import apply_migrations

    target = bind or engine
    Base.metadata.create_all(bind=target)
    result = apply_migrations(target)
    global _postgis_enabled
    if bind is None:
        _postgis_enabled = bool(result.get("postgis"))
    return result


def postgis_enabled() -> bool:
    return _postgis_enabled


def database_status() -> dict:
    info = parse_database_url(settings.database_url)
    return {
        "type": info["kind"],
        "dialect": info["dialect"],
        "safe_url": info["safe_url"],
        "postgis": postgis_enabled() if info["is_postgresql"] else False,
        "pool_size": None if info["is_sqlite"] else settings.db_pool_size,
        "sqlite_is_dev_fallback": info["is_sqlite"],
        "production_target": "postgresql+postgis",
    }
