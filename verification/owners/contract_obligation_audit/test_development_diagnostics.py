# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored diagnostic failure controls, never model training or audit examples."""
import copy
import struct
import unittest

from contracts import FOUNDATION, sha
from data import pair
from development_diagnostics import structure, wire_diagnostics


def rows(split):
    result = []
    for index, label in enumerate(("Entailment", "Contradiction")):
        text = split + " authored source " + str(index)
        result.append(pair({"split": split, "evidence": text, "evidence_sha256": sha(text.encode()),
                            "source_group": split + str(index), "hypothesis_id": "authored-hypothesis",
                            "hypothesis": "Recipient may retain a record.", "label": label}))
    return result


def bits(number):
    return struct.unpack("<I", struct.pack("<f", number))[0]


def wires(scores):
    value = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
             "tokenizer_hash": FOUNDATION["tokenizer.json"], "adapter_hash": "a" * 64,
             "results": [{"score": bits(score)} for score in scores]}
    return {style: copy.deepcopy(value) for style in ("json", "prose")}


class DevelopmentDiagnosticsTests(unittest.TestCase):
    def test_primary_path_preserves_both_formats_and_no_audit_claim(self):
        report = structure(rows("train"), rows("validation"), lambda text: [1, 2])
        self.assertFalse(report["independent_audit"])
        self.assertEqual(report["splits"]["validation"]["formats"]["prose"]["literal_answers_checked"], 4)
        self.assertEqual(report["splits"]["train"]["formats"]["json"]["maximum_tokens"], 2)

    def test_audit_rows_cannot_be_renamed_as_validation(self):
        with self.assertRaises(ValueError):
            structure(rows("train"), rows("audit"))

    def test_source_overlap_is_refused(self):
        validation = rows("validation")
        validation[0]["source_group"] = rows("train")[0]["source_group"]
        with self.assertRaises(ValueError):
            structure(rows("train"), validation)

    def test_stored_contrast_mutation_is_refused(self):
        validation = rows("validation")
        validation[0]["chosen"] = validation[0]["rejected"]
        with self.assertRaises(ValueError):
            structure(rows("train"), validation)

    def test_truncation_and_invalid_token_vectors_are_refused(self):
        for vector in ([1] * 513, [], [True], [50368]):
            with self.assertRaises(ValueError):
                structure(rows("train"), rows("validation"), lambda text: vector)

    def test_missing_label_is_refused(self):
        with self.assertRaises(ValueError):
            structure(rows("train"), rows("validation")[:1])

    def test_literal_wire_ties_and_format_disagreements_are_measured(self):
        wire = wires([1, 0, 0, 0])
        wire["prose"]["results"] = [{"score": bits(score)} for score in (0, 1, 1, 0)]
        report = wire_diagnostics(rows("validation"), wire, "a" * 64)
        self.assertEqual(report["formats"]["json"]["ties"], 1)
        self.assertEqual(report["format_sign_disagreements"], 2)
        self.assertFalse(report["new_inference_performed"])

    def test_wire_prefix_nonfinite_and_count_failures_are_refused(self):
        for mutation in ("prefix", "nonfinite", "count", "shape"):
            wire = wires([1, 0, 1, 0])
            if mutation == "prefix":
                wire["json"]["adapter_hash"] = "0x" + "a" * 64
            elif mutation == "nonfinite":
                wire["json"]["results"][0]["score"] = 0x7F800000
            elif mutation == "count":
                wire["json"]["results"].pop()
            else:
                wire["json"]["results"][0] = None
            with self.assertRaises(ValueError):
                wire_diagnostics(rows("validation"), wire, "a" * 64)


if __name__ == "__main__":
    unittest.main()
