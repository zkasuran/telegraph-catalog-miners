# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bind the isolated reference tool to freshly observed live family classes."""
import datetime as dt
import json
import urllib.request

from contracts import FAMILY, write_json

NODE = "https://devnode.telegraphprotocol.com"


def validate_classes(family, child):
    resolved = family.get("resolve", {})
    if resolved.get("intent") != FAMILY or resolved.get("class") != "judgment" or resolved.get("scorer") != "judgment" or resolved.get("routed") is not True:
        raise ValueError("live family classification is not the assigned routed judgment task")
    if child.get("intent_id") != FAMILY + "_GENERAL" or child.get("class") != "judgment" or child.get("evaluation", {}).get("kind") != "judgment":
        raise ValueError("live child classification disagrees with assigned judgment family")


def collect_classes(target):
    urls = {"family": NODE + "/api/wasm?intent=" + FAMILY + "&resolve=" + FAMILY,
            "child": NODE + "/engine/v1/intents/" + FAMILY + "_GENERAL"}
    values = {}
    for name, url in urls.items():
        request = urllib.request.Request(url, headers={"User-Agent": "Threat-IP-live-class-verification/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read((1 << 20) + 1)
            if len(raw) > 1 << 20 or response.url != url:
                raise ValueError("live class response ceiling or origin mismatch")
            values[name] = json.loads(raw)
    validate_classes(values["family"], values["child"])
    proof = {"schema_version": 1, "family": FAMILY, "child": FAMILY + "_GENERAL", "class": "judgment",
             "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "urls": urls,
             "family_resolve": values["family"]["resolve"], "child_class": values["child"]["class"],
             "child_evaluation": values["child"]["evaluation"], "controller_catalog_changed": False}
    write_json(target, proof)
    return proof
