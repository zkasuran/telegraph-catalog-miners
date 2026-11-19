# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import base64
import hashlib
import json
import os
import re
import struct
from datetime import datetime, timezone

REPO = "zkasuran/telegraph-catalog-miners"

NODE = "https://devnode.telegraphprotocol.com"

KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"

KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"

FOUNDATION_RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"

BASE_CONFIG_URL = "https://huggingface.co/answerdotai/ModernBERT-base/resolve/8949b909ec900327062f0ebf497f51aef5e6f0c8/config.json"

BASE_CONFIG_SHA = "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f"

FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}

HOST = "98.95.110.88"

HOST_KEY = "98.95.110.88 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMRYtmP2ESjH1ah2MI3eqUmEgtVYeyR81RAbr0MB/r0G\n"

DEFAULT_OWNER = "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287"

SSH_USER = "telegraph-verify"

WINDOW = 5 * 3600

MAX_BODY = 1 << 20

MAX_HEADERS = 16 << 10

MAX_BATCH = 100

MAX_TOKENS = 512

MAX_TOTAL_TOKENS = MAX_BATCH * MAX_TOKENS

VOCAB_SIZE = 50368

READ_TIMEOUT = 10

SCORE_TIMEOUT = 600

MAX_CONNECTIONS = 8

SHA = re.compile(r"[0-9a-f]{64}")

FAMILY = re.compile(r"[A-Z][A-Z0-9_]{1,79}")

COMMIT = re.compile(r"[0-9a-f]{40}")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()

def strict_json(raw):
    def fields(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=fields, parse_constant=invalid_constant)

def utc(value):
    if not isinstance(value, str):
        # Lease/decision parsers share one malformed-value exception contract.
        raise ValueError("timestamp must be a string")  # noqa: TRY004
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("UTC timestamp required")
    return parsed

def now():
    return datetime.now(timezone.utc)

def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("GITHUB_REPOSITORY") != REPO or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"):
        raise RuntimeError("candidate preparation and runtime are remote-only")

def immutable_url(url):
    prefix = "https://raw.githubusercontent.com/" + REPO + "/"
    if not isinstance(url, str) or not url.startswith(prefix) or len(url) > 512:
        raise ValueError("immutable public repository URL required")
    remainder = url[len(prefix):]
    revision, separator, path = remainder.partition("/")
    if (not COMMIT.fullmatch(revision) or not separator or not path or
            not re.fullmatch(r"[A-Za-z0-9_/.-]+", path) or
            any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("immutable URL must name an exact commit and safe path")
    return url

def public_key(value):
    if not isinstance(value, str) or not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: [A-Za-z0-9_-]{1,80})?", value):
        raise ValueError("invalid lease public key")
    blob = base64.b64decode(value.split()[1], validate=True)
    if len(blob) != 51 or blob[:19] != struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32):
        raise ValueError("invalid ed25519 public key wire shape")
    return " ".join(value.split()[:2])

def lease_for(row, key, batch, run_id, commit, created=None):
    created = created or now()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", batch) or not re.fullmatch(r"[1-9][0-9]{0,19}", run_id) or not COMMIT.fullmatch(commit):
        raise ValueError("invalid integration lease identity")
    key = public_key(key)
    expires = datetime.fromtimestamp(created.timestamp() + WINDOW, timezone.utc)
    restrictions = (f'restrict,port-forwarding,permitlisten="127.0.0.1:{row["forward_port"]}",'
                    f'permitopen="registration-denied.invalid:1",command="/bin/false",expiry-time="{expires.strftime("%Y%m%d%H%M%SZ")}"')
    return {"schema_version": 1, "family": row["family"], "owner": row["owner"], "slot": row["slot"],
            "run_id": run_id, "commit": commit, "batch": batch, "public_key": key,
            "authorized_keys_line": restrictions + " " + key, "adapter_hash": row["keccak"],
            "package_sha256": row["sha256"], "forward_port": row["forward_port"], "runnable_url": row["runnable_url"],
            "existing_registration_id": row["existing_registration_id"], "created_at": created.isoformat(),
            "expires_at": expires.isoformat(), "candidate_proof_sha256": row["candidate_proof"]["sha256"],
            "runtime_proof_sha256": row["runtime_proof"]["sha256"], "source_proof_sha256": row["source_proof"]["sha256"],
            "required_server_policy": {"AllowTcpForwarding": "remote", "AllowStreamLocalForwarding": "no",
                                       "PermitTunnel": "no", "GatewayPorts": "no"}}

def validate_lease(lease, row, clock=None):
    clock = clock or now()
    created = utc(lease["created_at"])
    expected = lease_for(row, lease["public_key"], lease["batch"], lease["run_id"], lease["commit"], created)
    if lease != expected or created > clock or clock >= utc(lease["expires_at"]):
        raise ValueError("lease identity, bounds or expiry mismatch")
    return created
