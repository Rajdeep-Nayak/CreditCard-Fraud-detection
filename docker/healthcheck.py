"""Container health probe: exit 0 if the API answers /health with 200."""

import os
import sys
import urllib.request

port = os.environ.get("PORT", "8000")
try:
    ok = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4).status == 200
except Exception:
    ok = False
sys.exit(0 if ok else 1)
