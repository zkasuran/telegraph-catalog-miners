# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Extract all released table record units without inventing specimen counts."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.bold_depth = 0
        self.sup_depth = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = {"line": self.getpos()[0], "cells": []}
        elif tag == "td":
            self.cell = {"parts": [], "bold": [], "footnotes": [], "attrs": dict(attrs)}
        elif tag == "b":
            self.bold_depth += 1
        elif tag == "sup":
            self.sup_depth += 1
        elif tag in ("p", "br"):
            self.handle_data("\n")

    def handle_endtag(self, tag):
        if tag == "b":
            self.bold_depth -= 1
        elif tag == "sup":
            self.sup_depth -= 1
        elif tag == "p":
            self.handle_data("\n")
        elif tag == "td":
            if self.cell is None or self.row is None:
                raise ValueError("invalid table cell nesting")
            self.row["cells"].append({
                "text": normalized("".join(self.cell["parts"])),
                "bold_text": normalized("".join(self.cell["bold"])),
                "footnotes": normalized(" ".join(self.cell["footnotes"])),
                "attrs": self.cell["attrs"],
            })
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            if self.sup_depth:
                self.cell["footnotes"].append(data)
            else:
                self.cell["parts"].append(data)
                if self.bold_depth:
                    self.cell["bold"].append(data)


def parse_table(text):
    parser = TableParser()
    parser.feed(text)
    return parser


def main():
    source = ROOT / "sources"
    receipts_path = ROOT / "SOURCE-RECEIPTS.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else json.loads((ROOT / "FROZEN-MANIFEST.json").read_text())["files"]
    for entry in receipts:
        if "sha256" not in entry:
            raise ValueError("incomplete source download")
        if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("source hash mismatch: " + entry["path"])
    article = ET.parse(source / "newton2008.xml").getroot()
    grant = "".join(article.find(".//license-p").itertext())
    assert "any lawful purpose" in grant
    supplements = [n.attrib["id"].split("-")[-1] for n in article.findall(".//supplementary-material")]
    assert sorted(supplements) == ["sd001", "sd002", "sd003", "sd004", "sg001", "st001", "st002"]
    for suffix in supplements:
        if suffix != "sg001":
            text = (source / f"newton2008-{suffix}.html").read_text()
            parser = parse_table(text)
            (source / f"newton2008-{suffix}.txt").write_text("".join(parser.text))
    parser = parse_table((source / "newton2008-st001.html").read_text())
    records = []
    sections = []
    excluded = []
    section = None
    for i, row in enumerate(parser.rows):
        cells = row["cells"]
        if len(cells) == 1 and "Samples" in cells[0]["text"]:
            section = cells[0]["text"]
            sections.append(section)
        if len(cells) != 7:
            excluded.append({"row": i, "line": row["line"], "cells": cells})
            continue
        if cells[0]["text"] == "Sample":
            excluded.append({"row": i, "line": row["line"], "reason": "header", "cells": cells})
            continue
        identity, batch, packaging, hplc, ms, xrd, botany = cells
        if not identity["text"]:
            excluded.append({"row": i, "line": row["line"], "reason": "blank identity", "cells": cells})
            continue
        label = "Counterfeit" if identity["bold_text"] else "Authentic" if "genuine" in packaging["text"].casefold() else "Inconclusive"
        limit = []
        if identity["text"].startswith("S18,"):
            limit.append("HPLC and mass spectrometry were from a physically identical package in the same stock, not S18")
        if "Samples, seized" in (section or ""):
            event = "china2007"
        else:
            event = "jupiter2005_2006"
        record = {
            "record_id": f"newton2008-st001-row{i}", "source_doi": "10.1371/journal.pmed.0050032.st001",
            "html_line": row["line"], "section": section, "acquisition_event": event,
            "source_label": label, "label_basis": "bold specimen identity means counterfeit; genuine packaging cell means source-authored genuine",
            "sample": identity["text"], "batch_dates": batch["text"], "packaging": packaging["text"],
            "hplc": hplc["text"], "mass_spectrometry": ms["text"], "xrd": xrd["text"], "botany": botany["text"],
            "footnotes": [c["footnotes"] for c in cells], "specific_limits": limit,
        }
        record["entity_group"] = "mekophar" if "mekophar" in identity["text"].casefold() else "traphaco" if "traphaco" in identity["text"].casefold() else "guilin-artesunate"
        numbers = re.findall(r"\d+", packaging["text"])
        record["counterfeit_family_group"] = None if label != "Counterfeit" else "linked-types-4-10-15" if set(numbers) & {"4", "10", "15"} else "type-" + numbers[0] if numbers else "unresolved-family"
        records.append(record)
        print(record["record_id"], label, identity["text"], "|", batch["text"], "|", packaging["text"], "|", hplc["text"])
    out = {
        "schema_version": 1, "source_grant": grant,
        "attribution": "Newton PN et al. (2008), PLoS Med 5(2): e32, doi:10.1371/journal.pmed.0050032",
        "paper_population_not_released_records": 391, "paper_detailed_samples": 49,
        "released_record_units": len(records), "all_table_rows": len(parser.rows),
        "label_counts": {k: sum(r["source_label"] == k for r in records) for k in ("Authentic", "Counterfeit", "Inconclusive")},
        "specimen_count_warning": "Some row identities represent multiple specimens. Never equate record units with individual specimens.",
        "custody_limit": "Ad hoc collection lacked a legal chain of evidence. Source labels only, not fresh-item certification.",
        "records": records, "excluded_table_rows": excluded,
    }
    (ROOT / "SOURCE-INVENTORY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("INVENTORY", len(records), out["label_counts"], "of", len(parser.rows), "table rows")


if __name__ == "__main__":
    main()
