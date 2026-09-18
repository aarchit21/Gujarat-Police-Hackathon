"""Prove the database can take one writer per camera worker.

Written for: anyone who needs evidence that raising the worker count is safe on
this host, and a way to re-check it after a configuration change.

Background: the recorded four-camera run produced 34 "database is locked"
failures, all of them SQLite write-lock contention on UPDATE cameras. That is
why the worker cap sat at 4. This script reproduces the same write pattern at a
chosen concurrency and reports how many writes failed.

    python scripts/verify_concurrency.py                  # configured database
    python scripts/verify_concurrency.py --workers 32
    python scripts/verify_concurrency.py --url sqlite:///data/cctv.db --workers 8

It writes only to the audit table, using a marker action, and deletes its own
rows afterwards unless --keep is given.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import delete, func, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import make_engine, make_session_factory, parse_database_url  # noqa: E402
from app.models import AuditEvent, Camera  # noqa: E402

MARKER = "concurrency_probe"


def worker(idx: int, Session, rounds: int, results: list, errors: list, started: threading.Barrier) -> None:
    latencies: list[float] = []
    started.wait()
    db = Session()
    try:
        for r in range(rounds):
            t0 = time.perf_counter()
            try:
                # Same shape as a camera worker: append a row and touch the
                # camera row, which is the UPDATE that used to deadlock.
                db.add(AuditEvent(actor=f"probe{idx}", action=MARKER, detail=f"{idx}:{r}"))
                camera = db.scalar(select(Camera).limit(1))
                if camera is not None:
                    camera.last_pts_ms = float(idx * 1000 + r)
                db.commit()
                latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:  # noqa: BLE001 - we are counting failures
                db.rollback()
                errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
    finally:
        db.close()
    results.extend(latencies)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=settings.database_url)
    ap.add_argument("--workers", type=int, default=int(settings.max_concurrent_workers))
    ap.add_argument("--rounds", type=int, default=25, help="commits per worker")
    ap.add_argument("--keep", action="store_true", help="do not delete the probe rows")
    args = ap.parse_args()

    info = parse_database_url(args.url)
    engine = make_engine(args.url)
    Session = make_session_factory(engine)

    with Session() as db:
        if db.scalar(select(func.count()).select_from(Camera)) == 0:
            print("no cameras in the database; run scripts/seed.py first", file=sys.stderr)
            return 1

    print(f"database : {info['safe_url']}")
    print(f"workers  : {args.workers} threads x {args.rounds} commits = {args.workers * args.rounds} writes")
    if not info["is_sqlite"]:
        print(f"pool     : {settings.effective_db_pool_size()} + {settings.effective_db_max_overflow()} overflow")

    results: list[float] = []
    errors: list[str] = []
    barrier = threading.Barrier(args.workers)
    threads = [
        threading.Thread(target=worker, args=(i, Session, args.rounds, results, errors, barrier), daemon=True)
        for i in range(args.workers)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    attempted = args.workers * args.rounds
    ok = len(results)
    print(f"\nsucceeded: {ok}/{attempted}")
    print(f"failed   : {len(errors)}")
    if results:
        results.sort()
        print(
            f"commit ms: median {statistics.median(results):.1f} "
            f"p95 {results[int(len(results) * 0.95) - 1]:.1f} max {results[-1]:.1f}"
        )
    print(f"wall     : {elapsed:.1f}s  ({ok / max(elapsed, 1e-6):.0f} commits/s)")

    if errors:
        from collections import Counter

        print("\nfailure kinds:")
        for kind, n in Counter(e.split(":")[0] for e in errors).most_common():
            print(f"  {n:>5}  {kind}")
        print(f"  example: {errors[0]}")

    if not args.keep:
        with Session() as db:
            db.execute(delete(AuditEvent).where(AuditEvent.action == MARKER))
            db.commit()
        print("\nprobe rows removed")

    if errors:
        print("\nRESULT: this database cannot take that many concurrent writers.")
        return 1
    print(f"\nRESULT: {args.workers} concurrent writers, no failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
