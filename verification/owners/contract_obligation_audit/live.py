# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Read actual public task/class contracts, never evaluation-pool content."""
import argparse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from contracts import FAMILY, strict_json, write_json

NODE = "https://devnode.telegraphprotocol.com"


def validate(snapshot):
    family = strict_json(snapshot["family_raw"])["resolve"]
    child = strict_json(snapshot["child_raw"])
    if family["intent"] != FAMILY or family["class"] != "judgment" or child["class"] != "judgment":
        raise ValueError("current family/child class mismatch")
    if child["intent_id"] != FAMILY + "_GENERAL":
        raise ValueError("current child identity mismatch")
    if child["request"]["params"][0]["name"] != "contract" or child["onchain"]["ready"] is not True:
        raise ValueError("current input or schema readiness mismatch")
    fields = {field["name"]: field for field in child["response"]["fields"]}
    if (fields["verdict"]["allowed"] != ["Compliant", "Partially compliant", "Non-compliant", "Unclear"]
            or fields["clauses"].get("off_chain") is not True or fields["obligations"].get("off_chain") is not True):
        raise ValueError("current literal response contract changed")
    return child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshot = {"fetched_at": datetime.now(timezone.utc).isoformat()}
    for key, url in (("family", NODE + "/api/wasm?resolve=" + FAMILY),
                     ("child", NODE + "/engine/v1/intents/" + FAMILY + "_GENERAL")):
        request = urllib.request.Request(url, headers={"User-Agent": "contract-obligation-live-class/1.0", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read((2 << 20) + 1)
        if len(raw) > 2 << 20:
            raise ValueError("public schema byte ceiling")
        snapshot[key + "_raw"] = raw.decode()
        snapshot[key + "_url"] = url
    validate(snapshot)
    write_json(args.out, snapshot)
    print("Live family, child and literal response schema verified")


if __name__ == "__main__":
    main()
