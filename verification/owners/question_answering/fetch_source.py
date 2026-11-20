# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import hashlib
import time
import urllib.request

MAX_DOWNLOAD_BYTES = 700 << 20

def fetch(url, path, expected, ceiling=MAX_DOWNLOAD_BYTES):
    if path.is_file() and path.stat().st_size <= ceiling:
        h = hashlib.sha256()
        with path.open('rb') as cached:
            for chunk in iter(lambda: cached.read(1 << 20), b''):
                h.update(chunk)
        if h.hexdigest() == expected:
            return
    partial = path.with_suffix(path.suffix + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'Telegraph-training-check/1.0'})
    started = time.monotonic()
    h, total = (hashlib.sha256(), 0)
    try:
        with urllib.request.urlopen(request, timeout=45) as response, partial.open('wb') as out:
            while (chunk := response.read(1 << 20)):
                total += len(chunk)
                if total > ceiling or time.monotonic() - started > 600:
                    raise ValueError('source download exceeded byte or time ceiling')
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected:
            raise ValueError('source hash mismatch: ' + path.name)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
