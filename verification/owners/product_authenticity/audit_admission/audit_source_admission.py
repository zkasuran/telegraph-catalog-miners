# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only audit admission. Never promote quality or lookup results to truth."""
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NIGERIA_PDF = "91ab6c464da40d8f9761d46504b9e9659dd33cd0328c542de25756b6866bd945"
BANGLADESH_XLSX = "fd8759f025daa914f2659ab6f82401ac99e433cda02e12198e38d253499e6a80"
LIMIT = 128 << 20
APIS = ("Hydrochlorothiazide", "Cefuroxime axetil", "Co\u2010trimoxazole*", "Ciprofloxacin",
        "Glibenclamide", "Dexamethasone", "Metronidazole", "Ceftriaxone", "Chloroquine",
        "Fluconazole", "Furosemide", "Metformin", "Atenolol")
QUALITY = r"(?:in specification|moderate deviation|extreme deviation|probably falsified|falsified)"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def checked(raw, expected):
    if len(raw) > LIMIT or sha(raw) != expected:
        raise ValueError("original source hash or size mismatch")
    return raw


def source_grant(raw):
    if len(raw) > 4 << 20 or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("source XML ceiling or declarations")
    root = ET.fromstring(raw)
    grants = [" ".join(n.itertext()).strip() for n in root.findall(".//permissions/license")]
    if len(grants) != 1 or "Creative Commons" not in grants[0] or "Attribution" not in grants[0]:
        raise ValueError("source-specific commercial grant absent")
    if re.search(r"non.?commercial|CC.?BY.?NC", grants[0], re.I):
        raise ValueError("noncommercial source refused")
    names = []
    for group in root.findall(".//contrib-group"):
        for node in group.findall("contrib"):
            if node.get("contrib-type", group.get("content-type")) == "author":
                surname = node.findtext(".//surname", "")
                given = node.findtext(".//given-names", "")
                if surname:
                    names.append((given + " " + surname).strip())
    if not names:
        raise ValueError("original author attribution absent")
    links = sorted({v for n in root.findall(".//permissions/license") for sub in n.iter()
                    for v in [sub.get("{http://www.w3.org/1999/xlink}href")]
                    if v and "creativecommons.org" in v})
    if not links:
        raise ValueError("original licence link absent")
    return {"source_xml_sha256": sha(raw), "grant": grants[0], "license_links": links, "creators": names}


def nigeria_rows(text):
    if len(text.encode()) > 2 << 20:
        raise ValueError("source text ceiling")
    # Every S2 row must be consumed, including split-batch suffixes and the skipped
    # QMN 196 identifier. Table order is not the original specimen identifier.
    start = text.index("Supplementary Table S2: List of all samples")
    end = text.index("Supplementary Table S3. Results of the examined PIN", start)
    lines = text[start:end].splitlines()
    records, pending = [], []
    for offset, line in enumerate(lines):
        code = re.search(r"QMN\s+(\d{3})(?:\s+(\([12]/2\)))?\s*$", line)
        if not code:
            pending.append(line)
            continue
        ordinal = re.match(r"\s*(\d+)\s+", line)
        result = re.search(r"(" + QUALITY + r")\s+QMN", line)
        country = re.search(r"\s+(India|Nigeria|China|UK|France|Spain|Malaysia|Thailand|missing)\s+([AB])\s+", line)
        if not ordinal or not result or not country:
            raise ValueError("unparsed full source row " + code.group(1))
        before = line[ordinal.end():country.start()]
        found = list(re.finditer("|".join(re.escape(a) for a in APIS), before))
        if not found:
            raise ValueError("unparsed stated API " + code.group(1))
        api = found[-1]
        product = before[:api.start()].strip()
        manufacturer = before[api.end():].strip()
        # Wrapped manufacturer text precedes the numbered row. Restrict recovery
        # to that column so registration-verification notes do not become names.
        column = ordinal.end() + api.end()
        while column < country.start() and line[column].isspace():
            column += 1
        prefix = []
        for prior in pending:
            origin = len(prior) - len(prior.lstrip())
            if origin < column - 4 or len(prior) <= column:
                continue
            part = prior[:country.start(1)].strip()
            if part:
                prefix.append(part)
        manufacturer = " ".join(prefix + [manufacturer]).strip()
        middle = line[country.end():result.start()]
        # Registration, lookup status, batch and method remain source text. Batch
        # is the final column before the method, not an invented acquisition ID.
        method = re.search(r"(\S+)\s+(HPLC; UV/Vis|HPLC|UV/Vis|\u2010|-)\s+" + QUALITY, middle)
        if not method:
            raise ValueError("unparsed source method " + code.group(1))
        batch = method.group(1)
        status = result.group(1)
        truth = "Counterfeit" if status == "falsified" else "Unknown"
        records.append({"source_ordinal": int(ordinal.group(1)), "record_id": "QMN " + code.group(1),
                        "split_batch_suffix": code.group(2), "product": product, "stated_api": api.group(),
                        "stated_manufacturer": manufacturer, "stated_country": country.group(1),
                        "source_vendor_type": country.group(2), "batch": batch,
                        "source_combined_quality": status, "physical_authenticity": truth,
                        "source_doi": "10.4269/ajtmh.23-0837", "source_text_line": len(text[:start].splitlines()) + offset + 1,
                        "raw_source_row": line.strip(), "vendor_identity_released": False,
                        "individual_collection_time_released": False,
                        "preference_provenance": None})
        pending = []
    if len(records) != 260 or [r["source_ordinal"] for r in records] != list(range(1, 261)):
        raise ValueError("incomplete 260-row source population")
    if len({normalized(r["record_id"]) for r in records}) != 260:
        raise ValueError("normalized specimen identity duplicate")
    if {r["record_id"] for r in records if r["physical_authenticity"] == "Counterfeit"} != {"QMN 015", "QMN 043", "QMN 045", "QMN 046"}:
        raise ValueError("original falsified identities changed")
    return records


def workbook(raw):
    checked(raw, BANGLADESH_XLSX)
    sheets, members = [], []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.infolist()
        if len(info) > 500 or sum(i.file_size for i in info) > LIMIT:
            raise ValueError("workbook expansion ceiling")
        if len({i.filename for i in info}) != len(info):
            raise ValueError("duplicate workbook member")
        for item in info:
            name = Path(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("workbook path escape")
            data = archive.read(item)
            members.append({"name": item.filename, "bytes": len(data), "sha256": sha(data)})
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        paths = {n.get("Id"): n.get("Target") for n in rel}
        for sheet in root.findall("s:sheets/s:sheet", NS):
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = paths[rid]
            path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = []
            for row in ET.fromstring(archive.read(path)).findall("s:sheetData/s:row", NS):
                cells = {}
                for cell in row.findall("s:c", NS):
                    value = cell.findtext("s:v", "", NS)
                    if cell.get("t") == "s":
                        value = strings[int(value)] if value else ""
                    elif cell.get("t") == "inlineStr":
                        value = "".join(cell.find("s:is", NS).itertext())
                    if value:
                        cells[cell.get("r")] = value
                if cells:
                    rows.append({"source_row": int(row.get("r")), "cells": cells})
            sheets.append({"name": sheet.get("name"), "state": sheet.get("state", "visible"), "path": path, "rows": rows})
    cells = [value for s in sheets for row in s["rows"] for value in row["cells"].values()]
    codes = sorted({m.group() for value in cells for m in re.finditer(r"\b(?:[ABCD]|AA|PS)-\d{3}\b", value)})
    return {"sha256": sha(raw), "members": members, "sheets": sheets, "specimen_codes_in_cells": codes,
            "contains_released_specimen_table": bool(codes)}


def admission(records, book, grants):
    if len(grants) != 2 or any(not g.get("creators") or not g.get("grant") or not g.get("license_links") for g in grants):
        raise ValueError("both original grants and attribution required")
    if any(r["physical_authenticity"] != ("Counterfeit" if r["source_combined_quality"] == "falsified" else "Unknown") for r in records):
        raise ValueError("quality/lookup/uncertain state promoted to physical truth")
    quality = Counter(r["source_combined_quality"] for r in records)
    batches = Counter((normalized(r["product"]), normalized(r["stated_manufacturer"]), normalized(r["batch"])) for r in records)
    return {"schema_version": 1, "family": "PRODUCT_AUTHENTICITY", "status": "AUDIT_ADMISSION_REFUSED",
            "source_scope": "Primary source inventory and specimen truth limits, not a model audit",
            "nigeria": {"released_specimens": len(records), "source_quality_counts": dict(quality),
                        "physical_authenticity_counts": dict(Counter(r["physical_authenticity"] for r in records)),
                        "reported_vendor_count": 62, "released_vendor_identities": 0,
                        "repeated_brand_manufacturer_batch_groups": sum(n > 1 for n in batches.values()),
                        "extra_units_in_repeated_batch_groups": sum(n - 1 for n in batches.values()),
                        "article_reported_repeated_batch_groups": 25,
                        "raw_name_grouping_is_not_independence_proof": True},
            "bangladesh": {"reported_sample_count": 189, "aggregate_manufacturer_confirmed_genuine": 40,
                           "manufacturers_confirming_genuine": 4, "released_workbook_sheets": len(book["sheets"]),
                           "sheet_names": [s["name"] for s in book["sheets"]],
                           "released_workbook_specimen_codes": book["specimen_codes_in_cells"],
                           "aggregate_confirmations_admitted_as_individual_labels": 0},
            "grants": grants, "certified_independent_audit_groups": 0,
            "required_independent_groups": 32, "required_audit_pairs": 64,
            "audit_pairs_created": 0, "model_operations": 0, "registration_ready": False,
            "limits": ["Nigeria Table S2 releases only vendor type A/B, not the 62 vendor identities or individual collection times.",
                       "Stated manufacturers of falsified products are not verified actual makers or independent counterfeiter families.",
                       "Nigeria in-specification and verified registration rows do not establish genuine physical specimens.",
                       "Bangladesh workbook contains analytical methods, not the 189 specimen records or 40 confirmation-to-specimen mappings.",
                       "Bangladesh's 40 confirmations come from four manufacturers, not 40 independent manufacturer groups.",
                       "Neither the aggregate populations nor repetitions can satisfy the independent audit gate."]}


def build(directory, output):
    directory, output = Path(directory), Path(output)
    pdf = checked((directory / "audit-source-files/nigeria-supplement.pdf").read_bytes(), NIGERIA_PDF)
    if not pdf.startswith(b"%PDF-"):
        raise ValueError("source PDF signature")
    text = (directory / "audit-source-files/nigeria-supplement.txt").read_text()
    records = nigeria_rows(text)
    book = workbook((directory / "audit-source-files/bangladesh-data.xlsx").read_bytes())
    grants = [source_grant((directory / (p + ".xml")).read_bytes()) for p in ("PMC11229646", "PMC8741935")]
    report = admission(records, book, grants)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"NIGERIA-SOURCE-RECORDS.json": {"records": records, "grant": grants[0], "pdf_sha256": sha(pdf), "text_sha256": sha(text.encode())},
                 "BANGLADESH-FULL-WORKBOOK.json": book, "AUDIT-ADMISSION.json": report}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return report


def snapshot(directory):
    """Verify the exact licensed safe projection in a standalone public checkout."""
    directory = Path(directory)
    raw = (directory / "SOURCE-MANIFEST.json").read_bytes()
    if len(raw) > 1 << 20:
        raise ValueError("manifest ceiling")
    manifest = json.loads(raw)
    names = {"nigeria-table-s2.txt", "bangladesh-workbook-summary.json", "source-grants.json"}
    if manifest.get("schema_version") != 1 or manifest.get("family") != "PRODUCT_AUTHENTICITY" or set(manifest.get("files", {})) != names:
        raise ValueError("standalone snapshot schema")
    data = {}
    for name in sorted(names):
        path = directory / name
        if path.is_symlink():
            raise ValueError("snapshot symlink")
        metadata = manifest["files"][name]
        raw = path.read_bytes()
        if len(raw) > 2 << 20 or len(raw) != metadata["bytes"] or sha(raw) != metadata["sha256"]:
            raise ValueError("frozen snapshot identity mismatch")
        data[name] = raw
    records = nigeria_rows(data["nigeria-table-s2.txt"].decode())
    book = json.loads(data["bangladesh-workbook-summary.json"])
    if book["sha256"] != BANGLADESH_XLSX or book["specimen_codes_in_cells"] != [] or len(book["sheets"]) != 3:
        raise ValueError("original workbook projection mismatch")
    grants = json.loads(data["source-grants.json"])
    if [g["source_xml_sha256"] for g in grants] != ["a5cf3728e8aeb6b33e391cb493fb937b7c869fe28ab847017ad14c081492fb39",
                                                    "a59480a14301be18e6635af0fcfb53447c17e2de209022c71e4a3720749556cd"]:
        raise ValueError("original article identity mismatch")
    report = admission(records, book, grants)
    report["safe_snapshot_manifest_sha256"] = sha((directory / "SOURCE-MANIFEST.json").read_bytes())
    report["commit"] = os.environ.get("GITHUB_SHA")
    report["run_id"] = int(os.environ["GITHUB_RUN_ID"]) if os.environ.get("GITHUB_RUN_ID") else None
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources")
    parser.add_argument("--output")
    parser.add_argument("--snapshot")
    args = parser.parse_args()
    if args.snapshot and not args.sources and not args.output:
        result = snapshot(args.snapshot)
    elif args.sources and args.output and not args.snapshot:
        result = build(args.sources, args.output)
    else:
        parser.error("use --snapshot or both --sources and --output")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
