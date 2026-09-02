"""Gujarat CCTV hybrid P0 — registry, ANPR sightings, watchlist alerts, GIS.

Force RTSP-over-TCP before any OpenCV VideoCapture import path.
"""

import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
