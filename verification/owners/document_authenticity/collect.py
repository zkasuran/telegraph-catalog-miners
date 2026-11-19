# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded source inventory and Poppler measurements, never neural work."""
import argparse
import datetime
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scorer import FIELDS, PACKAGE, compact, digest, make_pair

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "spec.json").read_text())
NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("source redirect refused")


class Fetcher:
    def __init__(self, root):
        self.root = root
        self.receipts = json.loads((root / "receipts.json").read_text()) if (root / "receipts.json").exists() else []
        self.total = sum(row["bytes"] for row in self.receipts)
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, url, name, ceiling):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {"www.govinfo.gov", "devnode.telegraphprotocol.com"} or parsed.username or parsed.password or parsed.query:
            raise ValueError("nonallowlisted source")
        if parsed.hostname == "devnode.telegraphprotocol.com" and parsed.path != "/engine/v1/intents/DOCUMENT_AUTHENTICITY_GENERAL":
            raise ValueError("only public family definition allowed; no pool")
        path = self.root / name
        # Resume only exact previously measured source bytes. Never overwrite them.
        if path.exists():
            matches = [r for r in self.receipts if r["path"] == name and r["url"] == url]
            if path.stat().st_size > ceiling or len(matches) != 1:
                raise ValueError("unbound cached source")
            raw = path.read_bytes()
            if digest(raw) != matches[0]["sha256"]:
                raise ValueError("cached source hash changed")
            return raw
        if len(self.receipts) >= SPEC["limits"]["get_count"]:
            raise ValueError("GET count ceiling")
        started = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": "DocumentEvidenceCheck/1.0", "Accept": "*/*"})
        with self.opener.open(request, timeout=SPEC["limits"]["get_seconds"]) as response:
            if response.status != 200:
                raise ValueError("source status not 200")
            raw = response.read(ceiling + 1)
            if len(raw) > ceiling or time.monotonic() - started > SPEC["limits"]["get_seconds"]:
                raise ValueError("source byte or time ceiling")
        self.total += len(raw)
        if self.total > SPEC["limits"]["total_bytes"]:
            raise ValueError("aggregate source ceiling")
        path.write_bytes(raw)
        self.receipts.append({"url": url, "path": name, "sha256": digest(raw), "bytes": len(raw),
                              "at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        save(self.root / "receipts.json", self.receipts)
        return raw


def sitemap(raw):
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("XML entities refused")
    tree = ET.fromstring(raw)
    if tree.tag not in (NS + "sitemapindex", NS + "urlset"):
        raise ValueError("sitemap root contract")
    return tree.tag == NS + "sitemapindex", [node.text for node in tree.iter(NS + "loc")]


def inventory(fetch):
    index, locations = sitemap(fetch.get("https://www.govinfo.gov/sitemap/BILLS_sitemap_index.xml", "index.xml", SPEC["limits"]["body_bytes"]))
    if not index or len(locations) != len(set(locations)):
        raise ValueError("collection index contract")
    expected = {f"https://www.govinfo.gov/sitemap/BILLS_{year}_sitemap.xml" for year in SPEC["inventory"]["years"]}
    if not expected.issubset(locations):
        raise ValueError("missing selected source year")
    packages, counts = {}, {}
    # Complete all selected session years, never a sorted prefix of one listing.
    for url in sorted(expected):
        index, rows = sitemap(fetch.get(url, url.rsplit("/", 1)[1], SPEC["limits"]["body_bytes"]))
        if index:
            raise ValueError("nested sitemap must be inventoried, not silently truncated")
        counts[url] = len(rows)
        for location in rows:
            prefix = "https://www.govinfo.gov/app/details/"
            if not isinstance(location, str) or not location.startswith(prefix):
                raise ValueError("unexpected package URL")
            package = location[len(prefix):]
            match = PACKAGE.fullmatch(package)
            if match and match[1] in SPEC["inventory"]["sessions"] and package not in SPEC["exclude"]:
                packages[package] = SPEC["inventory"]["sessions"][match[1]]
    selected = []
    for session, split in SPEC["inventory"]["sessions"].items():
        candidates = [p for p in packages if PACKAGE.fullmatch(p)[1] == session]
        candidates.sort(key=lambda p: digest(("document-owner-v1:" + p).encode()))
        if len(candidates) < SPEC["inventory"]["per_session"]:
            raise ValueError("insufficient full-session source inventory")
        selected.extend({"package": p, "split": split} for p in candidates[:SPEC["inventory"]["per_session"]])
    report = {"scope": SPEC["inventory"], "collection_index_maps": len(locations), "fully_read_maps": counts,
              "eligible_enrolled_packages": dict(Counter(packages.values())), "eligible_total": len(packages),
              "selected": selected, "population_claim": "Only selected sessions. Not all GovInfo or all BILLS."}
    save(fetch.root / "inventory.json", report)
    return selected


def parse_pdfsig(text):
    # Exactly one signature. Unknown output never becomes false or a label.
    if len(re.findall(r"^Signature #\d+:", text, re.M)) != 1:
        raise ValueError("expected one measured signature")
    status = re.findall(r"^  - Signature Validation: (.+)$", text, re.M)
    if status == ["Signature is Valid."]:
        integrity = True
    elif status == ["Digest Mismatch."]:
        integrity = False
    else:
        raise ValueError("unrecognized mathematical signature status: " + repr(status))
    whole, partial = "  - Total document signed", "  - Not total document signed"
    if (whole in text) == (partial in text):
        raise ValueError("unknown signed-range coverage")
    return {"integrity": integrity, "whole_document": whole in text, "trust": "unchecked"}


def measure(path):
    env = {"PATH": os.defpath, "LC_ALL": "C", "HOME": str(HERE)}
    result = subprocess.run(["pdfsig", str(path)], capture_output=True, text=True, env=env,
                            timeout=SPEC["limits"]["pdfsig_seconds"], check=False)
    if result.returncode != 0 or len(result.stdout.encode()) > 16384:
        raise ValueError("pdfsig failed or output ceiling")
    path.with_suffix(".pdfsig.txt").write_text(result.stdout)
    return parse_pdfsig(result.stdout)


def variants(raw):
    # Change only the signed PDF version digit. Page dictionaries may be in
    # compressed object streams. Header mutation needs no PDF content rewriting.
    match = re.match(rb"%PDF-1\.(\d)", raw)
    if not match:
        raise ValueError("no bounded PDF header mutation target")
    pos = match.start(1)
    ranges = re.findall(rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]", raw)
    if len(ranges) != 1:
        raise ValueError("expected one ByteRange")
    a, b, c, d = map(int, ranges[0])
    if not (a <= pos < a+b or c <= pos < c+d):
        raise ValueError("mutation outside signed bytes")
    tamper = raw[:pos] + (b"6" if raw[pos:pos+1] != b"6" else b"7") + raw[pos+1:]
    return {"original": raw, "header_tamper": tamper, "unsigned_padding": raw+b"\n",
            "tamper_and_padding": tamper+b"\n"}, pos


def collect(root):
    root.mkdir(parents=True, exist_ok=True)
    fetch = Fetcher(root)
    try:
        live = json.loads(fetch.get("https://devnode.telegraphprotocol.com/engine/v1/intents/DOCUMENT_AUTHENTICITY_GENERAL", "live.json", 262144))
        if live["class"] != "judgment" or live["intent_id"] != "DOCUMENT_AUTHENTICITY_GENERAL":
            raise ValueError("live family changed")
        fetch.get("https://www.govinfo.gov/about/policies", "policies.html", 1048576)
        fetch.get("https://www.govinfo.gov/about/authentication", "authentication.html", 1048576)
        selected = inventory(fetch)
        version = subprocess.run(["pdfsig", "-v"], capture_output=True, text=True, check=True, timeout=10)
        measurements, pairs, groups = [], [], set()
        for item in selected:
            package = item["package"]
            url = f"https://www.govinfo.gov/content/pkg/{package}/pdf/{package}.pdf"
            raw = fetch.get(url, package+".pdf", SPEC["limits"]["pdf_bytes"])
            if not raw.startswith(b"%PDF-"):
                raise ValueError("source is not a PDF")
            group = digest(raw)
            if group in groups:
                raise ValueError("duplicate original bytes")
            groups.add(group)
            views, position = variants(raw)
            for variant, content in views.items():
                path = root / (package + ("" if variant == "original" else "."+variant) + ".pdf")
                if variant != "original":
                    path.write_bytes(content)
                report = {"package": package, "sha256": digest(content), **measure(path)}
                expected = {"original": (True, True), "header_tamper": (False, True),
                            "unsigned_padding": (True, False), "tamper_and_padding": (False, False)}[variant]
                if tuple(report[f] for f in FIELDS) != expected:
                    raise ValueError("mutation expectation contradicted by actual pdfsig")
                measurements.append({**item, "group_id": group, "variant": variant, "report": report,
                                     "pdf": path.name, "pdfsig": path.with_suffix(".pdfsig.txt").name,
                                     "mutation_position": None if variant == "original" else position,
                                     "natural_forgery": False})
                pairs.extend(make_pair(report, field, group, item["split"], variant) for field in FIELDS)
            save(root / "measurements.json", measurements)
        for split in ("train", "validation", "audit"):
            rows = [row for row in pairs if row["split"] == split]
            (root / (split+".jsonl")).write_text("".join(compact(row)+"\n" for row in rows))
        save(root / "collection.json", {"spec_sha256": digest((HERE/"spec.json").read_bytes()),
                                       "originals": len(groups), "views": len(measurements), "pairs": len(pairs),
                                       "pdfsig_version": (version.stdout+version.stderr).strip(),
                                       "audit_status": "prepared, not model-scored or used to select a candidate"})
        print(compact({"originals": len(groups), "views": len(measurements), "pairs": len(pairs), "GETs": len(fetch.receipts)}))
    except Exception as exc:
        save(root / "failure.json", {"type": type(exc).__name__, "failure": str(exc), "receipts": len(fetch.receipts)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    collect(parser.parse_args().out)
