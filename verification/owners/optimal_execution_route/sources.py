# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import hashlib
import time
import urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
MAX_BYTES = 2 * 1024 * 1024

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def save_raw(raw):
    path = HERE / "evidence" / "raw" / (digest(raw) + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("content-addressed archive mismatch")
    else:
        path.write_bytes(raw)
    return str(path.relative_to(HERE))

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused")

def fetch(url, payload=None):
    req = urllib.request.Request(url, data=payload, headers={
        "User-Agent": "Telegraph-route-owner-source-verification/1",
        "Accept": "text/html" if "uniswap.org/" in url or "publicnode.com/terms" in url else "application/json",
        "Content-Type": "application/json",
    }, method="POST" if payload is not None else "GET")
    started = time.monotonic()
    with urllib.request.build_opener(NoRedirect()).open(req, timeout=20) as reply:
        raw = reply.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES or reply.status != 200:
            raise ValueError("source response limit or status")
    return raw, round(time.monotonic() - started, 3)
