# Image for the hosted demonstration instance.
#
# It carries the synthetic dataset from scripts/build_demo_dataset.py and
# NOTHING from the government cameras. .dockerignore denies everything by
# default and re-admits a short list, so the real corpus cannot arrive by
# someone forgetting to exclude a new directory.
#
#   python scripts/build_demo_dataset.py
#   docker build -t gp-console-demo .
#   docker run --rm -p 8000:8000 -e ADMIN_TOKEN=$(openssl rand -hex 16) gp-console-demo
#
# Verify before pushing:
#   docker run --rm gp-console-demo sh -c 'ls /app/data'          -> cctv.db, evidence
#   docker run --rm gp-console-demo sh -c 'grep -rl CSITMS /app'  -> nothing
FROM python:3.11-slim

# opencv-python-headless still links against glib even without a GUI. It is the
# only system package needed; the headless wheel avoids the whole Qt/libGL tree.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code edit does not reinstall 180 MB of wheels.
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY app/ ./app/
COPY scripts/__init__.py ./scripts/
# The demo dataset becomes the live dataset. Built outside the image because it
# needs Pillow, which the runtime image deliberately does not carry.
COPY data/demo/cctv.db ./data/cctv.db
COPY data/demo/evidence/ ./data/evidence/

ENV APP_ENV=production \
    ENABLE_DEVELOPER_UI=false \
    DEMO_INSTANCE=true \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ADMIN_TOKEN is deliberately NOT set here. The default in app/config.py is
# `p0-operator`, which is published in README.md -- it must be supplied by the
# host environment, not baked into a layer anyone can pull and read.

EXPOSE 8000

# Platform health check. /api/health needs a token and reports model names, the
# database type and the catalogue host; /healthz answers {"ok": true}.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/healthz', timeout=4).status==200 else 1)"

# Shell form so ${PORT} expands -- Render, Railway and Fly all inject it.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
