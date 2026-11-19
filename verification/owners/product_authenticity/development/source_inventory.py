# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import re
import xml.etree.ElementTree as ET
SANADA_DOI = '10.3390/pharmacy9010003'

def text(node):
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()

def sanada_records(xml):
    root = ET.fromstring(xml)
    paragraphs = [text(p) for p in root.findall(".//body//p")]
    materials = [p for p in paragraphs if "Authenticity was assessed by the manufacturer of the product." in p]
    if len(materials) != 1:
        raise ValueError("manufacturer label source paragraph absent or ambiguous")
    paragraph = materials[0]
    grants = [text(p) for p in root.findall(".//license-p")]
    if not any("Creative Commons Attribution (CC BY)" in p for p in grants):
        raise ValueError("source-specific CC BY grant absent")
    specs = [("Cialis", "20 mg", "GC", 9, "Authentic"), ("Cialis", "20 mg", "FC", 23, "Counterfeit"),
             ("Levitra", "20 mg", "GL", 9, "Authentic"), ("Levitra", "20 mg", "FL", 13, "Counterfeit"),
             ("Viagra", "100 mg", "GV", 4, "Authentic"), ("Viagra", "100 mg", "FV", 18, "Counterfeit")]
    records = []
    for brand, dose, prefix, count, label in specs:
        if not re.search(re.escape(prefix) + r"1\s*[–-]\s*" + re.escape(prefix) + str(count), paragraph):
            raise ValueError("published identifier range absent: " + prefix)
        for number in range(1, count + 1):
            sample = prefix + str(number)
            records.append({"record_id": "sanada2020-" + sample, "sample": sample, "brand": brand, "dose": dose,
                            "source_doi": SANADA_DOI, "source_label": label,
                            "label_basis": "Article reports manufacturer-assessed authenticity for this published identifier range.",
                            "source_paragraph": paragraph, "source_license": "CC-BY-4.0",
                            "cohort_group": "sanada-ed-internet-surveys", "entity_group": brand.casefold(),
                            "batch_group": "unreleased-batches-same-study", "raw_spectra_released": False})
    return records
