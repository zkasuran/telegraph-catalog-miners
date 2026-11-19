# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Public contracts for one bounded candidate integration lease."""
import base64
import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.request
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
ANCHORS = {
    "SENTIMENT_ANALYSIS": {
        "class": "judgment", "file": "SENTIMENT_ANALYSIS.safetensors",
        "sha256": "f2eefcc7b7fd807bcbd260d97c8b3b0a6b50276199b475131df968a5b60d43d9",
        "keccak": "4fba5da30996be940825450ae3abc5a438b1c582e098346469b83be42168a125",
        "slot": 1, "forward_port": 48601,
        "runnable_url": "https://tga1.zkasuran.dev/_s2sent/runnable",
        "verified_run_id": "44441475723", "verified_commit": "a18b51ce4be30741a86e2432416b5b283a795b4d",
        "candidate_sha256": "32c0e285f6726b07ced06f07b53f33a94194d373973de56bfb74f94c55ecfa7d",
        "runtime_sha256": "dd66981ce9936b2bf7f906a6be416ec4a3a6f3d4830bdfe5691b1778614b32a5",
        "source_sha256": "4ee10138832d9958f6e06f6da8c19eecedd1f10936d100e95e74def299e6e119",
    },
}


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


def write_json(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


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


def validate_row(value):
    required = {"family", "class", "file", "url", "sha256", "keccak", "slot", "forward_port",
                "runnable_url", "verified_run_id", "verified_commit", "candidate_proof", "runtime_proof", "source_proof"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"owner", "child", "existing_registration_id"}:
        raise ValueError("candidate registry fields mismatch")
    row = dict(value)
    family = row["family"]
    if not isinstance(family, str) or not FAMILY.fullmatch(family):
        raise ValueError("invalid candidate family")
    row.setdefault("child", family + "_GENERAL")
    row.setdefault("owner", DEFAULT_OWNER)
    row.setdefault("existing_registration_id", 0)
    if row["child"] != family + "_GENERAL" or row["class"] not in ("judgment", "hybrid"):
        raise ValueError("candidate class or child mismatch")
    if not isinstance(row["file"], str) or not re.fullmatch(r"[A-Z0-9_.-]+\.safetensors", row["file"]):
        raise ValueError("invalid candidate filename")
    for key in ("sha256", "keccak"):
        if not isinstance(row[key], str) or not SHA.fullmatch(row[key]):
            raise ValueError("invalid candidate hash")
    if not isinstance(row["owner"], str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", row["owner"]):
        raise ValueError("invalid candidate owner")
    row["owner"] = row["owner"].lower()
    if row["owner"] != DEFAULT_OWNER:
        raise ValueError("candidate owner is not the selected Telegraph identity")
    for key in ("slot", "forward_port", "existing_registration_id"):
        if type(row[key]) is not int:
            raise ValueError("candidate port, slot and id must be integers")
    if (not 1 <= row["slot"] <= 6 or row["forward_port"] != 48600 + row["slot"] or
            not 0 <= row["existing_registration_id"] < 1 << 63):
        raise ValueError("candidate port, slot or existing id mismatch")
    if not isinstance(row["runnable_url"], str) or not re.fullmatch(
            r"https://tga1\.zkasuran\.dev/_[a-zA-Z0-9_-]{1,64}/runnable", row["runnable_url"]):
        raise ValueError("public runnable URL mismatch")
    immutable_url(row["url"])
    if not isinstance(row["verified_run_id"], str) or not re.fullmatch(r"[1-9][0-9]{0,19}", row["verified_run_id"]):
        raise ValueError("invalid verified run")
    if not isinstance(row["verified_commit"], str) or not COMMIT.fullmatch(row["verified_commit"]):
        raise ValueError("invalid verified commit")
    for name in ("candidate", "runtime", "source"):
        proof = row[name + "_proof"]
        if not isinstance(proof, dict) or set(proof) != {"url", "sha256"}:
            raise ValueError("public proof URL and hash required")
        immutable_url(proof["url"])
        if not isinstance(proof["sha256"], str) or not SHA.fullmatch(proof["sha256"]):
            raise ValueError("invalid proof hash")
    # This campaign admits only reviewed anchors. Serving remains generic.
    anchor = ANCHORS.get(family)
    if anchor is None:
        raise ValueError("candidate has no verified campaign anchor")
    for key in ("class", "file", "sha256", "keccak", "slot", "forward_port", "runnable_url", "verified_run_id", "verified_commit"):
        if row[key] != anchor[key]:
            raise ValueError("verified candidate binding mismatch: " + key)
    for name in ("candidate", "runtime", "source"):
        if row[name + "_proof"]["sha256"] != anchor[name + "_sha256"]:
            raise ValueError("verified candidate proof binding mismatch")
    return row


def load_inventory(path, family, existing=0):
    if path.stat().st_size > MAX_BODY:
        raise ValueError("candidate inventory byte ceiling exceeded")
    value = strict_json(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {"schema_version", "candidates"} or value["schema_version"] != 1:
        raise ValueError("candidate inventory schema mismatch")
    if not isinstance(value["candidates"], list) or not 1 <= len(value["candidates"]) <= 6:
        raise ValueError("candidate inventory count mismatch")
    rows = [validate_row(row) for row in value["candidates"]]
    for field in ("family", "slot", "forward_port", "runnable_url"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError("duplicate candidate routing identity")
    matches = [row for row in rows if row["family"] == family]
    if len(matches) != 1:
        raise ValueError("unknown candidate family")
    row = matches[0]
    if type(existing) is not int or not 0 <= existing < 1 << 63:
        raise ValueError("invalid explicit existing registration id")
    if existing:
        row["existing_registration_id"] = existing
    return row


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_json(url, ceiling=MAX_BODY):
    if not url.startswith((NODE + "/", "https://api.github.com/repos/" + REPO + "/")):
        raise ValueError("public source GET URL outside allowlist")
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-candidate-integration/1.0", "Cache-Control": "no-cache"})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=25) as response:
        raw = response.read(ceiling + 1)
        if response.status != 200 or len(raw) > ceiling:
            raise ValueError("public GET status or byte ceiling exceeded")
    return strict_json(raw), raw


def fetch_hash(url, target, expected, ceiling, timeout=600):
    if not SHA.fullmatch(expected):
        raise ValueError("external file SHA-256 required")
    partial = target.with_name(target.name + ".partial")
    started = time.monotonic()
    total, digest = 0, hashlib.sha256()
    # All targets are immutable public raw, pinned kit or release URLs. No auth.
    allowed = ("https://raw.githubusercontent.com/" + REPO + "/", FOUNDATION_RELEASE,
               "https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT)
    if not url.startswith(allowed) and url != BASE_CONFIG_URL:
        raise ValueError("download URL outside public allowlist")
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-candidate-integration/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
            if response.status != 200 or not response.geturl().startswith("https://"):
                raise ValueError("download status or transport mismatch")
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - started > timeout:
                    raise ValueError("download byte or time ceiling exceeded")
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("download file hash mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def check_run(metadata, row):
    if (str(metadata.get("id")) != row["verified_run_id"] or metadata.get("head_sha") != row["verified_commit"] or
            metadata.get("conclusion") != "success" or metadata.get("status") != "completed" or
            metadata.get("path") != ".github/workflows/evaluator-sentiment-recover.yml" or
            metadata.get("repository", {}).get("full_name") != REPO or
            metadata.get("repository", {}).get("private") is not False or
            metadata.get("head_repository", {}).get("full_name") != REPO):
        raise ValueError("verified recovery run binding mismatch")


def check_proofs(row, candidate, runtime, source):
    identity = {"run_id": row["verified_run_id"], "commit": row["verified_commit"]}
    expected_package = {"file": row["file"], "sha256": row["sha256"], "keccak256": "0x" + row["keccak"], "bytes": 6798100}
    if (candidate.get("schema_version") != 1 or candidate.get("status") != "CANDIDATE_PENDING_CONTROLLER_REGISTRATION_TEST" or
            candidate.get("family") != row["family"] or candidate.get("child") != row["child"] or
            candidate.get("package") != expected_package or candidate.get("recovery_identity") != identity or
            candidate.get("kit_revision") != KIT or candidate.get("runtime_inputs") != FOUNDATION or
            candidate.get("retrained") is not False or candidate.get("originals") != 256):
        raise ValueError("candidate source/runtime identity mismatch")
    if (candidate.get("evidence_sha256", {}).get("runtime-result.json") != row["runtime_proof"]["sha256"] or
            candidate.get("evidence_sha256", {}).get("data-binding.json") != row["source_proof"]["sha256"]):
        raise ValueError("candidate proof lineage mismatch")
    training = candidate.get("training_identity", {})
    if (training.get("repository") != REPO or training.get("run_id") != 44266324976 or
            training.get("commit") != "a3a6f86c1bd831b41de8d99d12b0620c3a287f6a" or
            training.get("quality_report_sha256") != "4ffc26072a347b9af879afbd3c17fbecc1c1115bf6fe141943088577dff23422"):
        raise ValueError("original training identity mismatch")
    if (runtime.get("family") != row["family"] or runtime.get("verdict") != "PASS" or
            runtime.get("failures") != [] or runtime.get("recovery_identity") != identity or
            runtime.get("package_sha256") != row["sha256"] or runtime.get("adapter_hash") != "0x" + row["keccak"] or
            runtime.get("foundation_hash") != FOUNDATION["foundation_lora_fp32.onnx"] or
            runtime.get("engine_hash") != FOUNDATION["foundation_engine.wasm"] or
            runtime.get("tokenizer_hash") != FOUNDATION["tokenizer.json"] or
            runtime.get("originals") != 256 or runtime.get("repeat_bit_vectors") != 1536 or
            runtime.get("formats") != candidate.get("formats") or
            set(runtime.get("formats", {})) != {"json", "prose", "bare_label"}):
        raise ValueError("verified runtime proof mismatch")
    for metrics in runtime["formats"].values():
        scores = [metrics["pairwise_accuracy"], *metrics["accuracy_by_label"].values()]
        if (metrics["n_eval"] != 256 or metrics["repeat_bit_vectors"] != 512 or
                set(metrics["accuracy_by_label"]) != {"Positive", "Negative"} or
                any(type(v) not in (float, int) or not math.isfinite(v) for v in scores) or
                scores[0] < 0.80 or min(scores[1:]) < 0.70):
            raise ValueError("verified runtime quality gate mismatch")
    if (source.get("data_report_sha256") != "417f9901e5bfc7ddbff1b3cf016f973649f9c50ebf0410f2391583e80d214451" or
            source.get("normalized_source_overlap") != 0 or
            set(source.get("files", {})) != {"train", "validation", "test"}):
        raise ValueError("source-only proof boundary mismatch")
    for split, pairs in (("train", 768), ("validation", 128), ("test", 256)):
        file = source["files"][split]
        if file["pairs"] != pairs or file["labels"] != {"Positive": pairs // 2, "Negative": pairs // 2}:
            raise ValueError("source-only split count mismatch")


def check_classes(snapshot, row, clock=None):
    clock = clock or now()
    if set(snapshot) != {"schema", "fetched_at", "family", "child"} or snapshot["schema"] != 1:
        raise ValueError("class snapshot schema mismatch")
    age = (clock - utc(snapshot["fetched_at"])).total_seconds()
    if not -60 <= age <= 900:
        raise ValueError("stale class snapshot")
    parent, child = [], []
    for name, url in (("family", NODE + "/api/wasm?resolve=" + row["family"]),
                      ("child", NODE + "/engine/v1/intents/" + row["child"])):
        value = snapshot[name]
        if set(value) != {"url", "raw"} or value["url"] != url:
            raise ValueError("authoritative class URL mismatch")
        decoded = strict_json(value["raw"])
        if name == "family":
            parent = decoded.get("resolve", {})
        else:
            child = decoded
    if (parent.get("intent") != row["family"] or parent.get("class") != row["class"] or
            child.get("intent_id") != row["child"] or child.get("intent_name") != row["child"] or
            child.get("class") != row["class"] or child.get("evaluation", {}).get("kind") != row["class"]):
        raise ValueError("live family or class mismatch")
    if any(child.get(section, {}).get("served_by") != row["family"] for section in ("request", "response", "onchain")) or child.get("onchain", {}).get("ready") is not True:
        raise ValueError("live child schema family mismatch")


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
