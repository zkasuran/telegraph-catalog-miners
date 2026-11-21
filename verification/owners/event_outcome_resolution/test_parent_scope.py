# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic parent schema/record controls, not model validation results."""
import copy
from pathlib import Path
import types
import unittest

import parent_scope as scope
from common import sha, strict_json


class ParentScope(unittest.TestCase):
    def parent(self):
        return {"intent_id": scope.FAMILY, "intent_name": scope.FAMILY, "class": "judgment", "prefix": True,
                "request": {"served_by": scope.FAMILY}, "response": {"served_by": scope.FAMILY},
                "onchain": {"served_by": scope.FAMILY, "ready": True, "format_version": 4,
                            "request": copy.deepcopy(scope.REQUEST), "response": copy.deepcopy(scope.RESPONSE)}}

    def material(self, count=1):
        return {"intent_id": scope.FAMILY, "eligible_submitters": count, "min_submitters": 3,
                "active_snapshot": {"corpus_version": "a" * 64, "submitters": count,
                                    "rows": 150, "qualification": 78, "ranking": 72}}

    def test_parent_only_plan_primary_path_no_fake_metric(self):
        raw = Path(scope.__file__).with_name("frozen").joinpath("frozen.json").read_bytes()
        plan = scope.prepare(self.parent(), {"intent": scope.FAMILY, "class": "judgment"}, self.material(), strict_json(raw), sha(raw))
        self.assertEqual(plan["claimed_intents"], [scope.FAMILY])
        self.assertEqual(plan["training_record_intents"], [scope.FAMILY])
        self.assertFalse(plan["general_child_claimed"])
        self.assertFalse(plan["material"]["material_ready"])
        self.assertFalse(plan["registration_ready"])
        self.assertFalse(plan["training_record_emitted"])
        self.assertIsNone(plan["validation_accuracy"])

    def test_source_class_and_self_identity_are_not_overrides(self):
        for field, value in (("class", "hybrid"), ("intent_id", scope.FAMILY + "_GENERAL"), ("prefix", False)):
            parent = self.parent()
            parent[field] = value
            with self.assertRaises(ValueError):
                scope.parent_contract(parent, {"intent": scope.FAMILY, "class": "judgment"})

    def test_literal_wire_indexes_and_types(self):
        for value in (False, 0.0, "0"):
            parent = self.parent()
            parent["onchain"]["request"][0]["index"] = value
            with self.assertRaises(ValueError):
                scope.parent_contract(parent, {"intent": scope.FAMILY, "class": "judgment"})
        parent = self.parent()
        parent["onchain"]["response"][1]["type"] = "enum"
        with self.assertRaises(ValueError):
            scope.parent_contract(parent, {"intent": scope.FAMILY, "class": "judgment"})

    def test_child_status_never_substitutes_for_parent(self):
        status = self.material(8)
        status["intent_id"] += "_GENERAL"
        with self.assertRaises(ValueError):
            scope.material_contract(status)

    def test_material_floors_and_snapshot_counts_kept(self):
        self.assertFalse(scope.material_contract(self.material())["material_ready"])
        self.assertTrue(scope.material_contract(self.material(5))["material_ready"])
        status = self.material(8)
        status["active_snapshot"]["submitters"] = 1
        self.assertFalse(scope.material_contract(status)["material_ready"])

    def test_training_record_requires_metric_and_claims_only_parent(self):
        for value in (None, True, float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                scope.training_record(192, value)
        record = scope.training_record(192, 0.8)  # Synthetic protocol measurement only.
        self.assertEqual([item["intent"] for item in record["children"]], [scope.FAMILY])

    def test_output_cannot_mutate_literal_wire_contract(self):
        result = scope.parent_contract(self.parent(), {"intent": scope.FAMILY, "class": "judgment"})
        result["request"][0]["index"] = 99
        self.assertEqual(scope.REQUEST[0]["index"], 0)

    def test_actual_parent_identity_mutation_caught(self):
        source = Path(scope.__file__).read_text()
        mutant = types.ModuleType("mutant_parent_scope")
        exec(compile(source.replace('parent.get("intent_id") != FAMILY or ', ''), "mutant_parent_scope", "exec"), mutant.__dict__)
        parent = self.parent()
        parent["intent_id"] += "_GENERAL"
        with self.assertRaises(AssertionError):
            with self.assertRaises(ValueError):
                mutant.parent_contract(parent, {"intent": scope.FAMILY, "class": "judgment"})


if __name__ == "__main__":
    unittest.main()
