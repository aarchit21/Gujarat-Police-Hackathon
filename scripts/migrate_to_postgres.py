"""Copy the SQLite development database into PostgreSQL.

Written for: whoever moves this host off the SQLite fallback so many camera
workers can run at once. SQLite's whole-file write lock is what produced the
"database is locked" failures in the recorded four-camera run; it is a
development fallback, not a target for a catalogue of cameras.

The copy goes through SQLAlchemy Core against the shared metadata, so JSON
columns, booleans and timestamps are converted by the column types rather than
by hand. Tables are copied parents-first so foreign keys hold, and identity
sequences are advanced past the highest copied id.

Read-only with respect to the source: nothing is written back to SQLite.

    python scripts/migrate_to_postgres.py \
        --source sqlite:///data/cctv.db \
        --target postgresql+psycopg://cctv@127.0.0.1:5432/cctv

Add --dry-run to print row counts and stop. Re-running is safe: --truncate
clears the target tables first, otherwise existing rows are kept and the copy
is refused if the target already holds data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, func, insert, select, text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Base, make_engine, parse_database_url  # noqa: E402
from app import models as _models  # noqa: F401,E402  (registers the tables)

# Parents before children. Anything not named here is copied afterwards in
# metadata order, which SQLAlchemy already sorts by dependency.
PRIORITY = (
    "cameras",
    "watchlist",
    "system_state",
    "audit",
    "camera_activity",
    "vehicle_observations",
    "sightings",
    "alerts",
    "recognition_attempts",
)

BATCH = 500


def ordered_tables() -> list:
    by_name = {t.name: t for t in Base.metadata.sorted_tables}
    out = [by_name[n] for n in PRIORITY if n in by_name]
    out += [t for t in Base.metadata.sorted_tables if t.name not in PRIORITY]
    return out


def count_rows(engine, table) -> int:
    with engine.connect() as conn:
        try:
            return int(conn.execute(select(func.count()).select_from(table)).scalar_one())
        except Exception:
            return 0


def reset_sequences(engine, tables) -> list[str]:
    """Advance PostgreSQL identity sequences past the copied ids."""
    done = []
    with engine.begin() as conn:
        for table in tables:
            pk = list(table.primary_key.columns)
            if len(pk) != 1:
                continue
            column = pk[0]
            if not column.autoincrement or column.type.python_type is not int:
                continue
            seq = conn.execute(
                text("SELECT pg_get_serial_sequence(:t, :c)"),
                {"t": table.name, "c": column.name},
            ).scalar()
            if not seq:
                continue
            conn.execute(
                text(
                    f"SELECT setval(:seq, COALESCE((SELECT MAX({column.name}) FROM {table.name}), 0) + 1, false)"
                ),
                {"seq": seq},
            )
            done.append(table.name)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=f"sqlite:///{(ROOT / 'data' / 'cctv.db').as_posix()}")
    parser.add_argument("--target", required=True, help="postgresql+psycopg://user@host:port/db")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--truncate", action="store_true", help="clear target tables before copying")
    args = parser.parse_args()

    src_info = parse_database_url(args.source)
    tgt_info = parse_database_url(args.target)
    if not tgt_info["is_postgresql"]:
        raise SystemExit(f"--target must be PostgreSQL, got {tgt_info['safe_url']}")
    print(f"source {src_info['safe_url']}\ntarget {tgt_info['safe_url']}\n")

    source = create_engine(args.source, future=True)
    target = make_engine(args.target)

    tables = ordered_tables()
    counts = {t.name: count_rows(source, t) for t in tables}
    total = sum(counts.values())
    for name, n in counts.items():
        if n:
            print(f"  {name:<24} {n:>8}")
    print(f"  {'TOTAL':<24} {total:>8}\n")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    # Create the schema, run the same migrations the app runs at startup.
    from app.migrate import apply_migrations

    Base.metadata.create_all(bind=target)
    migrated = apply_migrations(target)
    print(f"schema ready (postgis={migrated.get('postgis')})")

    if args.truncate:
        with target.begin() as conn:
            names = ", ".join(t.name for t in reversed(tables))
            conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        print("target tables truncated")
    else:
        occupied = {t.name: count_rows(target, t) for t in tables}
        busy = {k: v for k, v in occupied.items() if v}
        if busy:
            raise SystemExit(
                f"target already holds rows {busy}; re-run with --truncate to replace them"
            )

    copied = 0
    for table in tables:
        if not counts.get(table.name):
            continue
        with source.connect() as sconn:
            rows = sconn.execute(select(table)).mappings()
            batch: list[dict] = []
            written = 0
            with target.begin() as tconn:
                for row in rows:
                    batch.append(dict(row))
                    if len(batch) >= BATCH:
                        tconn.execute(insert(table), batch)
                        written += len(batch)
                        batch = []
                if batch:
                    tconn.execute(insert(table), batch)
                    written += len(batch)
        copied += written
        print(f"  copied {table.name:<24} {written:>8}")

    fixed = reset_sequences(target, tables)
    print(f"\nsequences advanced: {', '.join(fixed) or 'none'}")

    verify = {t.name: count_rows(target, t) for t in tables}
    bad = {n: (counts[n], verify[n]) for n in counts if counts[n] != verify[n]}
    print(f"copied {copied} rows")
    if bad:
        print(f"MISMATCH (source, target): {bad}")
        return 1
    print("row counts match on every table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
