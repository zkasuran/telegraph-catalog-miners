# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only secondary block controls. Constructed generation is fixture-only."""
import copy
import ast
import json
import os
import unittest
import types
from unittest.mock import patch

from common import HERE, sha, audit_excerpts
from origin_contract import length_accuracy, fit_length
from secondary_contract import expanded_source_bundle, validate_secondary, secondary_preferences, attributions
from source_preferences import preference_rows
from test_source_preferences import population


class SecondaryTests(unittest.TestCase):
    def test_audit_source_identity_function_is_not_shadowed_by_wire_identity(self):
        import audit_runtime
        self.assertNotIn("identity", audit_runtime.main.__code__.co_varnames + audit_runtime.main.__code__.co_cellvars)
        self.assertIn("wire_identity", audit_runtime.main.__code__.co_varnames)
        self.assertEqual(audit_runtime.identity("  A B "), audit_runtime.identity("a b"))
        broken = (HERE / "audit_runtime.py").read_text().replace("wire_identity", "identity")
        main = next(n for n in ast.parse(broken).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        code = compile(ast.Module(body=[main], type_ignores=[]), "broken-audit-scope-fixture", "exec")
        function_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
        self.assertIn("identity", function_code.co_varnames + function_code.co_cellvars)

    def fixture(self):
        bundle, sources, _, base = expanded_source_bundle()
        records = []
        for row in sources["articles"]:
            fake = " ".join("f" + row["group_id"][:8] + str(i) for i in range(24))
            human, machine, _ = audit_excerpts(row["human_text"], fake)
            source = {"source_id": row["source_id"], "group_id": row["group_id"], "split": "train",
                "year": row["year"], "source_sha256": row["sha256"], "fixture_only": True}
            records += [{**source, "origin": "human_control", "text": human},
                {**source, "origin": "synthetic", "text": machine, "generator": "gpt2",
                 "generator_weight_sha256": base["generators"]["gpt2"]["weight_sha256"]}]
        raw = "".join(json.dumps(r) + "\n" for r in records).encode()
        report = {"output_sha256": sha(raw), "source_bundle_sha256": bundle,
            "secondary_source_manifest_sha256": sha((HERE / "secondary-training-sources.json").read_bytes()), "audit_used": False}
        return raw, report

    def test_sources_are_independent_of_all_old_and_retired_entities(self):
        _, sources, first, base = expanded_source_bundle()
        old = [r for rs in base["source_report"]["articles"].values() for r in rs] + first["articles"]
        self.assertEqual(len(sources["articles"]), 64)
        ids = {r["source_id"] for r in old} | set(base["audit_refresh"]["retired_article_ids"])
        authors = {a for r in old for a in r["author_ids"]} | set(base["audit_refresh"]["retired_author_ids"])
        for row in sources["articles"]:
            self.assertNotIn(row["source_id"], ids)
            self.assertFalse(authors.intersection(row["author_ids"]))
            self.assertIn("unrestricted", row["grant_clause"])
            self.assertIn("credited", row["grant_clause"])
            self.assertEqual(row["split"], "train")

    def test_fixture_generation_preserves_roles_and_length_control(self):
        raw, report = self.fixture()
        views = validate_secondary(raw, report, [])
        self.assertEqual(len(views), 128)
        self.assertEqual(length_accuracy(views, fit_length(views)), 0.5)
        with self.assertRaises(ValueError):
            validate_secondary(raw + b"changed", report, [])
        with self.assertRaises(ValueError):
            validate_secondary(raw, {**report, "audit_used": True}, [])
        with self.assertRaises(ValueError):
            validate_secondary(raw, report, [{"text_id": views[0]["text_id"]}])

    def test_fixture_secondary_preferences_are_additive_not_replacements(self):
        train, validation, first_views = population()
        first_pairs, _ = preference_rows(first_views, train, validation)
        raw, report = self.fixture()
        second = validate_secondary(raw, report, [])
        before = copy.deepcopy(train + first_pairs)
        pairs, contract = secondary_preferences(second, before)
        self.assertEqual(before, train + first_pairs)
        self.assertEqual(contract["combined_preferences"], 1008)
        self.assertEqual(contract["independent_training_groups"], 256)
        self.assertTrue(contract["repetition"]["passed"])
        self.assertTrue(all(r["style"] in ("json", "prose") for r in pairs))
        self.assertFalse(contract["audit_used"])

    def test_secondary_generator_refuses_local_execution(self):
        import generate_secondary
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                generate_secondary.main()

    def test_compact_attribution_retains_exact_original_grants(self):
        _, sources, _, _ = expanded_source_bundle()
        rows = attributions(sources["articles"])
        for compact, original in zip(rows, sources["articles"]):
            self.assertEqual(compact["grant_clause"], original["grant_clause"])
            self.assertEqual(compact["authors"], original["authors"])
            self.assertEqual(compact["sha256"], original["sha256"])
            self.assertNotIn("paragraph", compact)


if __name__ == "__main__":
    unittest.main()
