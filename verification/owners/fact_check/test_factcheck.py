# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Adversarial artificial schema fixtures only. No fixture enters the source corpus."""
import copy
import json
import os
import shutil
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import source_pack
from data import applicable, check_splits, digest, pair, source_assertion
from licensing import licence_metadata, preserve_artifact, verify_checkpoint_metadata
from preserve import preserve
from run import quality, remote_only
from runtime_check import check_response, decode_bits, native_agreement
from training_policy import development_selection

HERE = Path(__file__).resolve().parent


def statement(ident="Q1", object_id="Q2", rank="normal", qualifiers=None):
    return {"id": ident + "$fixture", "rank": rank, "mainsnak": {"property": "P36", "snaktype": "value",
        "datavalue": {"value": {"id": object_id}}}, "qualifiers": qualifiers or {},
        "references": [{"hash": "fixture-reference", "snaks": {"P248": [{"snaktype": "value", "datavalue": {"value": {"id": "Q9"}}}]}}]}


def fixture(statements=None):
    entity = {"id": "Q1", "lastrevid": 123, "labels": {"en": {"value": "Artificial region"}},
              "claims": {"P36": [statement()] if statements is None else statements}}
    objects = {"Q2": {"lastrevid": 124, "labels": {"en": {"value": "Artificial city"}}},
               "Q3": {"lastrevid": 125, "labels": {"en": {"value": "Other fixture city"}}}}
    return entity, objects


def time_snak(value, precision=11):
    return {"snaktype": "value", "datavalue": {"value": {"time": "+" + value + "T00:00:00Z",
            "precision": precision, "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
            "before": 0, "after": 0, "timezone": 0}}}


class SourceTests(unittest.TestCase):
    def test_source_statement_preserves_full_identity_reference_qualifier(self):
        entity, objects = fixture()
        row = source_assertion(entity, objects, {"sha256": "fixture-only"})
        self.assertEqual(row["entity_revision"], 123)
        self.assertEqual(row["active_statements"][0]["references"], entity["claims"]["P36"][0]["references"])
        self.assertEqual(row["active_statements"][0]["qualifiers"], {})

    def test_multiple_capitals_are_excluded_never_false(self):
        entity, objects = fixture([statement(), statement(object_id="Q3")])
        with self.assertRaisesRegex(ValueError, "multiple applicable"):
            source_assertion(entity, objects, {})

    def test_deprecated_statement_does_not_supply_a_correct_answer(self):
        entity, objects = fixture([statement(rank="deprecated")])
        with self.assertRaises(ValueError):
            source_assertion(entity, objects, {})

    def test_unknown_somevalue_reference_and_revision_fail_closed(self):
        for change in ("unknown", "reference", "revision", "qualifier", "capital_revision"):
            entity, objects = fixture()
            if change == "unknown":
                entity["claims"]["P36"][0]["mainsnak"]["snaktype"] = "somevalue"
            elif change == "reference":
                entity["claims"]["P36"][0]["references"] = []
            elif change == "revision":
                entity.pop("lastrevid")
            elif change == "qualifier":
                entity["claims"]["P36"][0]["qualifiers"] = {"P518": []}
            else:
                objects["Q2"].pop("lastrevid")
            with self.subTest(change=change), self.assertRaises(ValueError):
                source_assertion(entity, objects, {})

    def test_calendar_precision_and_transition_day(self):
        self.assertTrue(applicable(statement(qualifiers={"P580": [time_snak("2020-01-01")]})))
        self.assertFalse(applicable(statement(qualifiers={"P582": [time_snak("2020-01-01")]})))
        for snak in (time_snak("2020-01-01", 9), time_snak("2026-11-18")):
            with self.assertRaises(ValueError):
                applicable(statement(qualifiers={"P580": [snak]}))

    def test_false_claim_cannot_name_any_non_deprecated_alternative(self):
        entity, objects = fixture()
        row = source_assertion(entity, objects, {})
        wrong = pair(row, {"object_id": "Q3", "object_label": "Other fixture city"}, False)
        check_splits({"train": [wrong]})
        wrong["claim_object_id"] = "Q2"
        with self.assertRaises(ValueError):
            check_splits({"train": [wrong]})

    def test_entity_relation_and_variants_stay_in_one_split(self):
        entity, objects = fixture()
        row = pair(source_assertion(entity, objects, {}), {"object_id": "Q2", "object_label": "Artificial city"}, True)
        with self.assertRaisesRegex(ValueError, "cross-split"):
            check_splits({"train": [row], "audit": [copy.deepcopy(row)]})


class RuntimeGuards(unittest.TestCase):
    def test_environment_negatives_are_isolated_under_real_gha(self):
        good = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners",
                "GITHUB_EVENT_NAME": "workflow_dispatch", "RUNNER_TEMP": "/tmp/gha-fixture",
                "FACTCHECK_REPOSITORY_PRIVATE": "false", "RUNNER_ENVIRONMENT": "github-hosted"}
        with patch.dict(os.environ, good, clear=True):
            remote_only()
        for key in good:
            with self.subTest(key=key), patch.dict(os.environ, {**good, key: "wrong" if key != "RUNNER_TEMP" else ""}, clear=True), self.assertRaises(RuntimeError):
                remote_only()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            remote_only()

    def test_quality_thresholds_fail_independently(self):
        trained = {"pairwise_accuracy": .9, "subsets": {"chosen_longer": .9, "chosen_shorter": .9}}
        baseline = {"pairwise_accuracy": .5}
        controls = {"length": .5, "answer_only": .5}
        self.assertTrue(quality(trained, baseline, controls)["passed"])
        for value in (.79, float("nan")):
            self.assertFalse(quality({**trained, "pairwise_accuracy": value}, baseline, controls)["passed"])
        self.assertFalse(quality(trained, {"pairwise_accuracy": .88}, controls)["passed"])
        self.assertFalse(quality(trained, baseline, {"length": .66})["passed"])
        self.assertFalse(quality({**trained, "subsets": {"chosen_longer": .69}}, baseline, controls)["passed"])

    def test_dev_loss_selects_epoch_not_audit_accuracy(self):
        best, _ = development_selection(None, 1, {"val_loss": .3, "pairwise_accuracy": .99})
        best, improved = development_selection(best, 2, {"val_loss": .2, "pairwise_accuracy": .1})
        self.assertTrue(improved)
        self.assertEqual(best["epoch"], 2)
        best, improved = development_selection(best, 3, {"val_loss": .2})
        self.assertFalse(improved)

    def test_nonfinite_bits_and_wrong_identity_are_refused(self):
        with self.assertRaises(ValueError):
            decode_bits(0x7F800000)
        with self.assertRaises(ValueError):
            check_response({"results": []}, 1, {"keccak256": "0" * 64})

    def test_native_runtime_tolerance_is_not_cross_machine_bit_equality(self):
        native = [1.00001, 0.0] * 64
        bits = [struct.unpack("<I", struct.pack("<f", value))[0] for value in [1.0, 0.0] * 64]
        result = native_agreement(native, bits)
        self.assertTrue(result["numerical"] and result["ordering"])
        self.assertFalse(result["exact_bits_claimed"])


class RightsTests(unittest.TestCase):
    def test_exact_primary_archive_replays_without_existing_source_cache(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            for name in ("sources.tar.gz", "sources.json"):
                shutil.copyfile(HERE / name, root / name)
            with patch.object(source_pack, "HERE", root):
                source_pack.unpack()
            index = json.loads((root / "sources/index.json").read_text())
            self.assertTrue(index["complete"])
            for record in index["records"]:
                self.assertEqual(digest((root / "sources" / record["file"]).read_bytes()), record["sha256"])

    def test_failed_checkpoint_always_contains_original_grants(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            preserve(root)
            for name in ("initial", "checkpoint", "adapter"):
                self.assertEqual((root / name / "SOURCE-CC0.txt").read_bytes(), (HERE / "SOURCE-CC0.txt").read_bytes())
            self.assertIn("x-source-cc0", json.loads((root / "result.json").read_text())["licence_metadata"])

    def test_container_terms_leave_artificial_payload_unchanged(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "fixture.safetensors"
            header = json.dumps({"__metadata__": {}, "fixture": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"test")
            preserve_artifact(path)
            self.assertTrue(path.read_bytes().endswith(b"test"))
            verify_checkpoint_metadata(path)
            archive = Path(temp) / "fixture.pt"
            with zipfile.ZipFile(archive, "w") as out:
                out.writestr("fixture/data.pkl", b"artificial-not-pickle")
                out.writestr("fixture/data/0", b"payload")
            preserve_artifact(archive)
            verify_checkpoint_metadata(archive)
            self.assertIn("x-source-cc0", licence_metadata())


if __name__ == "__main__":
    unittest.main()
