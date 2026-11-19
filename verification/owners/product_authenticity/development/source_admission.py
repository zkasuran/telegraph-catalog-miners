# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Admit primary specimen annotations without promoting uncertain authenticity."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DOI = "10.1007/s43441-021-00262-3"
EXPECTED_SOURCE_SHA = "1ae21544043dd0f631ac94d91882eb682126a8fffebbac0fffd304f1e3298a0e"


def text(node):
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def table_grid(table):
    """Expand JATS row/column spans. Keep every source row and cell position."""
    carry = {}
    rows = []
    for tr in table.findall(".//tr"):
        cells = {column: value for column, (remaining, value) in carry.items()}
        next_carry = {column: (remaining - 1, value) for column, (remaining, value) in carry.items() if remaining > 1}
        column = 0
        for cell in tr:
            if cell.tag not in {"td", "th"}:
                continue
            while column in cells:
                column += 1
            colspan, rowspan = int(cell.get("colspan", "1")), int(cell.get("rowspan", "1"))
            if not 1 <= colspan <= 20 or not 1 <= rowspan <= 100:
                raise ValueError("table span ceiling")
            value = text(cell)
            for offset in range(colspan):
                if column + offset in cells:
                    raise ValueError("overlapping table spans")
                cells[column + offset] = value
                if rowspan > 1:
                    next_carry[column + offset] = (rowspan - 1, value)
            column += colspan
        rows.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
        carry = next_carry
    if carry:
        raise ValueError("table span runs beyond released rows")
    return rows


def admission_label(source_label, proposed_label, grant, observation_kind="natural_specimen"):
    if grant != "CC-BY-4.0":
        raise ValueError("compatible commercial source grant not verified")
    if observation_kind != "natural_specimen":
        raise ValueError("surrogate or constructed material is not natural counterfeit ground truth")
    mapping = {"Genuine": "Authentic", "Unknown": "Inconclusive"}
    if source_label not in mapping or proposed_label != mapping[source_label]:
        raise ValueError("proposed authenticity contradicts the source annotation")
    return mapping[source_label]


def extract(data):
    if hashlib.sha256(data).hexdigest() != EXPECTED_SOURCE_SHA:
        raise ValueError("primary source hash mismatch")
    root = ET.fromstring(data)
    grant = [text(n) for n in root.findall(".//license-p")]
    if not any("Creative Commons Attribution 4.0 International License" in value for value in grant):
        raise ValueError("article-specific commercial grant absent")
    tables = {n.get("id"): table_grid(n) for n in root.findall(".//table-wrap")}
    if set(tables) != {"Tab1", "Tab2", "Tab3", "Tab4"}:
        raise ValueError("incomplete released table inventory")
    rows = tables["Tab1"]
    if len(rows) != 41 or rows[0][0] != "Sample No" or rows[0][-1] != "Authenticity":
        raise ValueError("released specimen table shape changed")
    annotations = []
    for row in rows[1:]:
        if len(row) != 9:
            raise ValueError("row span expansion did not preserve all specimen fields")
        sample_no, site_no = int(row[0]), int(row[1])
        source_label = row[8]
        mapped = "Authentic" if source_label == "Genuine" else "Inconclusive"
        verdict = admission_label(source_label, mapped, "CC-BY-4.0")
        annotations.append({"record_id": f"metformin2021-sample{sample_no}", "source_sample_no": sample_no,
                            "source_site_no": site_no, "product": row[2], "dose_units": row[3],
                            "packaging_language": row[4], "labelled_country": row[5], "shipping_country": row[6],
                            "package_insert": row[7], "source_label": source_label, "diagnostic_verdict": verdict,
                            "source_doi": DOI, "source_license": "CC-BY-4.0",
                            "source_group": DOI, "entity_group": normalize(row[2]),
                            "site_group": f"metformin-japan2017-site{site_no}",
                            "acquisition_time_group": "japan-metformin-online-january2017",
                            "annotation_provenance": "Published source authors report manufacturer/distributor assessment or unresolved authenticity, not AI-created specimen labels."})
    if {r["source_sample_no"] for r in annotations} != set(range(1, 41)):
        raise ValueError("missing or duplicate specimen numbers")
    labels = Counter(r["source_label"] for r in annotations)
    sites = {r["source_site_no"] for r in annotations}
    assert labels == {"Genuine": 7, "Unknown": 33} and sites == set(range(1, 25))
    contributors = root.findall(".//contrib-group[@content-type='author']/contrib")
    if not contributors:
        contributors = root.findall(".//contrib[@contrib-type='author']")
    authors = [{"family": n.findtext(".//surname"), "given": n.findtext(".//given-names")}
               for n in contributors if n.find(".//surname") is not None]
    if {n["family"] for n in authors} != {"Zhu", "Yoshida", "Tsuboi", "Matsushita", "Kimura"} or len(authors) != 5:
        raise ValueError("original source creator attribution is missing or changed")
    return {"schema_version": 1, "source_doi": DOI, "primary_sha256": EXPECTED_SOURCE_SHA,
            "attribution": authors, "source_grants": grant, "source_annotation_license": "CC-BY-4.0",
            "modification": "Technical table-span expansion and source-verdict mapping. Original annotations retained.",
            "table_inventory": {key: len(value) for key, value in tables.items()}, "supplements": len(root.findall(".//supplementary-material")),
            "source_samples": len(annotations), "source_sites": len(sites), "source_label_counts": dict(labels),
            "records": annotations}


def audit_admission(dataset):
    labels = Counter(r["diagnostic_verdict"] for r in dataset["records"])
    site_groups = {r["site_group"] for r in dataset["records"]}
    identities = [normalize(r["record_id"]) for r in dataset["records"]]
    return {"schema_version": 1, "source_doi": DOI, "audit_admitted": False, "training_rows": 0,
            "audit_pairs": 0, "model_operations": 0, "potential_rows_not_an_audit": len(dataset["records"]),
            "source_sites_upper_bound": len(site_groups), "required_independent_audit_groups": 32,
            "normalized_duplicate_record_ids": len(identities) - len(set(identities)),
            "diagnostic_verdict_counts": dict(labels), "verified_counterfeit_annotations": labels["Counterfeit"],
            "reason": "24 source sites are below the 32 independent-group gate even before brand/batch/entity linking. This source has zero verified counterfeit annotations; unknown and quality failures cannot supply the missing label."}
