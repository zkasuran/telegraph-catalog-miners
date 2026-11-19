# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""The same token-only public proxy audited by runtime_audit.py."""
import argparse
import json
from pathlib import Path
from proxy import BoundedServer, Backend
from remote import remote_only

if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", type=int, required=True)
    parser.add_argument("--row", type=Path, required=True)
    args = parser.parse_args()
    with BoundedServer(("127.0.0.1", args.port), json.loads(args.row.read_text()), Backend(args.backend)) as server:
        server.serve_forever()
