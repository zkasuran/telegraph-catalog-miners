# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Archive the failed exposed freeze byte for byte before new source review."""
import json
import re
from pathlib import Path

from source_probe import sha, write

HERE = Path(__file__).resolve().parent
FAILED_SHA = "ca65b547545bee9858f69bc73a70c30695db947c5ba80c793ca9944e428212c0"
ARCHIVE = HERE / "archive" / ("failed-" + FAILED_SHA[:12])


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    names = ("pilot-manifest.json", "pilot-checks.json", "semantic-review.json", "rights.json", "observations.json")
    if sha(HERE / names[0]) != FAILED_SHA:
        raise ValueError("active pilot is not the exact exposed failed freeze")
    inventory = {}
    for name in names:
        source, target = HERE / name, ARCHIVE / name
        raw = source.read_bytes()
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("archive already exists with different bytes")
        target.write_bytes(raw)
        if sha(target) != sha(source):
            raise ValueError("failed archive copy changed bytes")
        inventory[name] = sha(target)
    manifest = json.loads((ARCHIVE / "pilot-manifest.json").read_text())
    review = json.loads((ARCHIVE / "semantic-review.json").read_text())
    issues = {"failed_links": review["known_failures"], "unresolved_context_flags": [],
              "status": "FAILED_EXPOSED_ARCHIVE_NEVER_VALID", "ai_review": True,
              "policy": "All archived components are excluded from the next pool, including links without a listed error. Context flags are exclusions, not proof that every flagged translation is wrong."}
    for split, blocks in manifest["splits"].items():
        for block in blocks:
            for e, f in block["links"]:
                et, ft = (manifest["originals"][str(sid)]["text"] for sid in (e, f))
                if re.search(r"\b(you|your|yours|they|them|their|it|its)\b", et, re.IGNORECASE):
                    issues["unresolved_context_flags"].append({"ids": [e, f], "old_split": split,
                        "english": et, "french": ft, "reason": "English pronoun number, referent or possessive context must not be silently specialized in a bidirectional reference."})
    write(ARCHIVE / "issues-and-ambiguities.json", issues)
    inventory["issues-and-ambiguities.json"] = sha(ARCHIVE / "issues-and-ambiguities.json")
    write(ARCHIVE / "ARCHIVE.json", {"status": "FAILED_EXPOSED_ARCHIVE", "files": inventory,
          "manifest_sha256": FAILED_SHA, "model_training_ever_started": False,
          "known_link_errors": len(review["known_failures"]), "unresolved_context_flags": len(issues["unresolved_context_flags"]),
          "source_rights": "Original CC BY 2.0 FR text and per-sentence attribution unchanged; full source grants retained in rights.json."})
    print(json.dumps({"archive": str(ARCHIVE), "known_link_errors": len(review["known_failures"]),
                      "unresolved_context_flags": len(issues["unresolved_context_flags"])}))


if __name__ == "__main__":
    main()
