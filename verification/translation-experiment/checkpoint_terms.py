# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Family-specific terms for both candidate and failed checkpoint containers."""
import base64
import hashlib
import importlib.util
import json
import os
import zlib
from pathlib import Path

from source_probe import sha

HERE = Path(__file__).resolve().parent
MAX_ATTRIBUTION_BYTES = 2 << 20


def encode_attribution(manifest):
    raw = json.dumps(manifest, ensure_ascii=False).encode()
    if len(raw) > MAX_ATTRIBUTION_BYTES:
        raise ValueError("complete attribution manifest exceeds its decoded ceiling")
    return json.dumps({"encoding": "zlib-base64", "sha256": hashlib.sha256(raw).hexdigest(),
                       "uncompressed_bytes": len(raw), "data": base64.b64encode(zlib.compress(raw, 9)).decode()})


def decode_attribution(value):
    envelope = json.loads(value)
    if envelope.get("encoding") != "zlib-base64" or type(envelope.get("uncompressed_bytes")) is not int or not 0 < envelope["uncompressed_bytes"] <= MAX_ATTRIBUTION_BYTES:
        raise ValueError("invalid attribution encoding or size")
    encoded = base64.b64decode(envelope["data"], validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(encoded, MAX_ATTRIBUTION_BYTES + 1)
    if (len(raw) != envelope["uncompressed_bytes"] or len(raw) > MAX_ATTRIBUTION_BYTES or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail or hashlib.sha256(raw).hexdigest() != envelope["sha256"]):
        raise ValueError("complete attribution manifest hash or decompression boundary failed")
    return json.loads(raw)


def family_bundle(rights, manifest):
    if manifest["source_license"] != "CC-BY-2.0-FR":
        raise ValueError("unreviewed source grant cannot enter this checkpoint policy")
    originals = manifest["originals"]
    if not originals or any(row["license"] != "CC-BY-2.0-FR" or not row["credited_contributor"]
                            or not row["sentence_url"].startswith("https://tatoeba.org/en/sentences/show/")
                            or row["original_text_changed"] for row in originals.values()):
        raise ValueError("source attribution, grant or original-text binding missing")
    # Source rights and full notices precede our terms. The shared blanket source
    # notice names other families and must never be substituted for this bundle.
    return {"tatoeba-source-rights": json.dumps(rights, ensure_ascii=False),
            "tatoeba-sentence-attribution": encode_attribution(manifest),
            "attribution-read-instructions": "Decode the zlib-base64 envelope with checkpoint_terms.decode_attribution. The envelope pins SHA-256 and decoded byte length. A plain complete pilot-manifest.json accompanies both successful and failed artifact downloads. Source grant terms remain uncompressed and readable.",
            "original-source-rights-retained": "CC BY 2.0 FR grants remain in force for original sentence text and links. Our SAND terms apply only to our additions. No endorsement.",
            "source-final-audit": "Source pool reviewed by AI before new deterministic freeze. A saved checkpoint is not a qualified translation scorer until all model, full parser and reference-runtime gates pass."}


def stamp(raw, bundle):
    spec = importlib.util.spec_from_file_location("unchanged_checkpoint_stamp", HERE.parent / "training/preserve.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.stamp(raw, bundle)


def remote_guard():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or not os.environ.get("RUNNER_TEMP")
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("Checkpoint publication runs only in an explicitly enabled bounded public CPU workflow")
    report = json.loads((HERE / "pilot-checks.json").read_text())
    if (not report["workflow_ready"] or report["semantic_review"] != "PASS"
            or report["pilot_manifest_sha256"] != sha(HERE / "pilot-manifest.json")
            or report["pilot_manifest_sha256"] != "099fa6f2c0f4eda56dd995ed5265e820b8c885bae58667e0e72f12db1f7e6f7b"):
        raise RuntimeError("Translation source/data review has not cleared")


if __name__ == "__main__":
    remote_guard()
