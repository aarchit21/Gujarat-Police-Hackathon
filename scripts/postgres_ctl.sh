#!/usr/bin/env bash
# Start, stop or check the user-space PostgreSQL cluster this project uses.
#
# Written for: anyone on the team who needs the database up before running the
# app or the camera workers. No root, no systemd -- PostgreSQL lives in its own
# conda environment so the solver can never disturb the `gujhac` ML stack, and
# the data directory sits outside the repository.
#
#   scripts/postgres_ctl.sh start|stop|status|psql|logs
#
# First-time setup is scripts/setup_postgres.py.
set -euo pipefail

PG_ENV="${PG_ENV:-$HOME/miniconda3/envs/pgsrv}"
PGDATA="${PGDATA:-$HOME/guj_pol_archit/pgdata}"
PGBIN="$PG_ENV/bin"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-cctv}"
PGDB="${PGDB:-cctv}"

if [ ! -x "$PGBIN/pg_ctl" ]; then
  echo "PostgreSQL not found at $PGBIN" >&2
  echo "Run: conda create -n pgsrv -c conda-forge -y postgresql" >&2
  exit 1
fi

case "${1:-status}" in
  start)
    if "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
      echo "already running"
    else
      "$PGBIN/pg_ctl" -D "$PGDATA" -l "$PGDATA/server.log" -w start
    fi
    "$PGBIN/psql" -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -tAc \
      "select 'ready: ' || current_database() || ' @ ' || version();"
    ;;
  stop)
    "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop
    ;;
  status)
    "$PGBIN/pg_ctl" -D "$PGDATA" status || true
    "$PGBIN/psql" -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -tAc \
      "select 'connections: ' || count(*) || '/' || current_setting('max_connections') from pg_stat_activity;" 2>/dev/null || true
    ;;
  psql)
    shift
    exec "$PGBIN/psql" -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d "$PGDB" "$@"
    ;;
  logs)
    tail -n "${2:-40}" "$PGDATA/server.log"
    ;;
  *)
    echo "usage: $0 start|stop|status|psql|logs" >&2
    exit 2
    ;;
esac
