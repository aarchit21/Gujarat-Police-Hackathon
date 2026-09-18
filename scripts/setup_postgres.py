"""One-time PostgreSQL setup for this project. Idempotent.

Written for: a teammate setting the project up on a new machine, or re-creating
the database after a wipe.

Why PostgreSQL: SQLite takes a whole-file write lock, which is why running four
camera workers produced "database is locked" failures. Reading a catalogue of
cameras at once needs a database that allows concurrent writers.

Why a separate conda environment: the ML environment (`gujhac`) holds a pinned
torch/opencv/onnxruntime stack. Installing a database server into it would let
the solver move those pins. PostgreSQL therefore lives in `pgsrv` and is reached
over TCP by psycopg, which is already installed in `gujhac`.

Why a user-space cluster: no root on this host, and the data directory stays
outside the repository so it is never committed or wiped by a clean checkout.

    python scripts/setup_postgres.py            # create cluster + database
    python scripts/setup_postgres.py --start    # ... and start the server

Afterwards, put the printed DATABASE_URL in .env and manage the server with
scripts/postgres_ctl.sh.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
DEFAULT_ENV = HOME / "miniconda3" / "envs" / "pgsrv"
DEFAULT_PGDATA = HOME / "guj_pol_archit" / "pgdata"

# Sized for many concurrent camera workers plus API traffic. The app opens one
# pooled connection per worker; see settings.effective_db_pool_size().
TUNING = """
# --- Gujarat CCTV platform: many concurrent camera workers ---
listen_addresses = '127.0.0.1'
port = {port}
max_connections = 200
shared_buffers = 512MB
work_mem = 16MB
maintenance_work_mem = 128MB
synchronous_commit = off
wal_compression = on
"""

MARKER = "Gujarat CCTV platform"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pg-env", type=Path, default=DEFAULT_ENV, help="conda env holding the postgres binaries")
    ap.add_argument("--pgdata", type=Path, default=DEFAULT_PGDATA)
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="cctv")
    ap.add_argument("--db", default="cctv")
    ap.add_argument("--start", action="store_true", help="start the server when setup finishes")
    args = ap.parse_args()

    pgbin = args.pg_env / "bin"
    if not (pgbin / "initdb").exists():
        print(f"PostgreSQL binaries not found in {pgbin}", file=sys.stderr)
        print("Create the server environment first:", file=sys.stderr)
        print("  conda create -n pgsrv -c conda-forge -y postgresql", file=sys.stderr)
        return 1

    version = run([str(pgbin / "postgres"), "--version"]).stdout.strip()
    print(f"using {version}")

    if (args.pgdata / "PG_VERSION").exists():
        print(f"cluster already initialised at {args.pgdata}")
    else:
        args.pgdata.mkdir(parents=True, exist_ok=True)
        # trust auth is acceptable because the server only listens on loopback.
        proc = run([
            str(pgbin / "initdb"), "-D", str(args.pgdata),
            "-U", args.user, "--auth=trust", "--encoding=UTF8",
        ])
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        print(f"cluster initialised at {args.pgdata}")

    conf = args.pgdata / "postgresql.conf"
    if MARKER not in conf.read_text():
        conf.write_text(conf.read_text() + TUNING.format(port=args.port))
        print("tuning appended to postgresql.conf")
    else:
        print("tuning already present")

    running = run([str(pgbin / "pg_ctl"), "-D", str(args.pgdata), "status"]).returncode == 0
    if args.start and not running:
        proc = run([
            str(pgbin / "pg_ctl"), "-D", str(args.pgdata),
            "-l", str(args.pgdata / "server.log"), "-w", "start",
        ])
        print(proc.stdout.strip() or proc.stderr.strip())
        running = proc.returncode == 0

    if running:
        env = {**os.environ, "PGPASSWORD": ""}
        exists = run([
            str(pgbin / "psql"), "-h", "127.0.0.1", "-p", str(args.port), "-U", args.user,
            "-d", "postgres", "-tAc", f"select 1 from pg_database where datname='{args.db}'",
        ], env=env).stdout.strip()
        if exists != "1":
            proc = run([
                str(pgbin / "createdb"), "-h", "127.0.0.1", "-p", str(args.port),
                "-U", args.user, args.db,
            ], env=env)
            print(f"database {args.db} created" if proc.returncode == 0 else proc.stderr.strip())
        else:
            print(f"database {args.db} already exists")
    elif not args.start:
        print("server not started (pass --start, or use scripts/postgres_ctl.sh start)")

    url = f"postgresql+psycopg://{args.user}@127.0.0.1:{args.port}/{args.db}"
    print(f"\nPut this in .env:\n  DATABASE_URL={url}")
    print("\nThen import the existing SQLite data if you have any:")
    print(f"  python scripts/migrate_to_postgres.py --target '{url}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
