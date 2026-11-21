# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure plan/geometry/guard controls; no local neural diagnostic measurements."""
import ast
import copy
import json
import os
import unittest
from unittest.mock import patch

from common import HERE, sha
from gradient_contract import diagnostic_policy, select_views, geometry, final_status, require_passed_inventory
from test_source_preferences import population
from source_preferences import preference_rows
from secondary_contract import validate_secondary, secondary_preferences
from origin_contract import equal_byte_views


def training_population():
    train, validation, first_views = population()
    first, _ = preference_rows(first_views, train, validation)
    raw = (HERE / "recovery/locked-secondary.jsonl").read_bytes()
    producer = json.loads((HERE / "recovery/locked-secondary-producer.json").read_text())
    secondary_views = validate_secondary(raw, producer, equal_byte_views(train) + equal_byte_views(validation) + first_views)
    recipe = json.loads((HERE / "recipe.json").read_text())
    secondary, _ = secondary_preferences(secondary_views, train + first, bare_groups=recipe["secondary_bare_groups"])
    return train + first + secondary


class GradientDiagnosticTests(unittest.TestCase):
    def test_fixed_training_plan_binds_exact_existing_rows(self):
        rows = training_population()
        recipe = json.loads((HERE / "recipe.json").read_text())
        selected = select_views(rows, diagnostic_policy(recipe))
        plan = json.loads((HERE / "gradient-diagnostic-plan.json").read_text())
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({g[0]["group_id"] for g in selected}), 6)
        self.assertEqual(plan["view_ids"], [g[0]["view_id"] for g in selected])
        self.assertEqual(plan["selected_rows_sha256"], sha(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()))
        self.assertEqual(plan["training_rows_sha256"], sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()))
        self.assertEqual([g[0]["style"] for g in selected], ["json", "json", "prose", "prose", "bare_label", "bare_label"])

    def test_validation_or_source_role_mutation_is_refused(self):
        rows = training_population()
        policy = diagnostic_policy(json.loads((HERE / "recipe.json").read_text()))
        bad = copy.deepcopy(rows)
        bad[0]["split"] = "validation"
        with self.assertRaises(ValueError):
            select_views(bad, policy)
        bad = copy.deepcopy(rows)
        bad[0]["group_id"] = "f" * 64
        with self.assertRaises(ValueError):
            select_views(bad, policy)

    def test_diagnostic_cannot_optimize_select_or_register(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        for key, value in (("optimizer_updates", 1), ("audit_used", True), ("validation_used", True),
                           ("gradient_surgery", True), ("registration_eligible", True), ("maximum_views", 64)):
            bad = copy.deepcopy(recipe)
            bad["gradient_diagnostic"][key] = value
            with self.assertRaises(ValueError):
                diagnostic_policy(bad)

    def test_gradient_geometry_positive_opposed_zero_and_invalid(self):
        self.assertEqual(geometry(1, 1, 1)["cosine"], 1)
        self.assertEqual(geometry(1, 1, -1)["cosine"], -1)
        self.assertTrue(geometry(1, 1, -1)["opposed"])
        self.assertIsNone(geometry(0, 1, 0)["cosine"])
        for values in ((-1, 1, 0), (1, 1, 2), (0, 1, 1), (float("nan"), 1, 0)):
            with self.assertRaises(ValueError):
                geometry(*values)

    def test_neural_probe_refuses_local_execution(self):
        import train_gradient_diagnostic
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_gradient_diagnostic.probe(*([None] * 9))

    def test_diagnostic_branch_precedes_optimizer_and_validation_measurement(self):
        tree = ast.parse((HERE / "train_formats.py").read_text())
        fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
        branch = next(n for n in fit.body if isinstance(n, ast.If) and "OWNER_DIAGNOSTIC_ONLY" in ast.unparse(n.test))
        optimizer = next(n for n in fit.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "optimizer" for t in n.targets))
        measure = next(n for n in fit.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "starting" for t in n.targets))
        self.assertLess(branch.lineno, optimizer.lineno)
        self.assertLess(branch.lineno, measure.lineno)
        self.assertTrue(any(isinstance(n, ast.Return) for n in branch.body))

    def test_diagnostic_has_no_optimizer_step_or_backward_call(self):
        tree = ast.parse((HERE / "train_gradient_diagnostic.py").read_text())
        forbidden = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"step", "backward"}}
        self.assertFalse(forbidden)
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "grad" for n in ast.walk(tree)))

    def test_direct_gap_and_original_tolerances_have_mutation_controls(self):
        source = (HERE / "train_gradient_diagnostic.py").read_text()
        def check(raw):
            tree = ast.parse(raw)
            gap = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "counterpart_gap")
            expression = next(n.value for n in gap.body if isinstance(n, ast.Return))
            if ast.unparse(expression) != "0.25 * torch.abs(a - b)":
                raise ValueError("exact direct gap required")
            for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "allclose"]:
                values = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
                if values.get("atol") != 2e-6 or values.get("rtol") != 2e-5:
                    raise ValueError("original gradient/value tolerances changed")
        check(source)
        for bad in (source.replace("0.25 * torch.abs(a - b)", "0.25 * torch.abs(a + b)"),
                    source.replace("atol=2e-6", "atol=2e-4")):
            with self.assertRaises(ValueError):
                check(bad)

    def test_remote_algebra_controls_refuse_local_neural_execution(self):
        import train_gradient_diagnostic
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_gradient_diagnostic.smoke(None)
            with self.assertRaises(RuntimeError):
                train_gradient_diagnostic.counterpart_gap(None, None)

    def test_added_reconstruction_cannot_override_original_refusal(self):
        views = [{"component_sum_gradient_passed": True, "score_seed_sum_passed": True,
                  "score_reconstruction_gradient_passed": True} for _ in range(6)]
        passed = {"status": final_status([], 6), "refusals": [], "views": views}
        require_passed_inventory(passed)
        bad = copy.deepcopy(passed)
        bad["views"][4]["component_sum_gradient_passed"] = False
        with self.assertRaises(ValueError):
            require_passed_inventory(bad)
        bad["refusals"] = [{"raw_gradient_sum_refused": True}]
        self.assertEqual(final_status(bad["refusals"], 6), "GRADIENT_DIAGNOSTIC_FAILED")
        with self.assertRaises(ValueError):
            require_passed_inventory(bad)
        with self.assertRaises(ValueError):
            final_status([], 4)

    def test_completed_inventory_is_saved_before_fatal_check(self):
        tree = ast.parse((HERE / "train_gradient_diagnostic.py").read_text())
        probe = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "probe")
        calls = [n.value for n in probe.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        self.assertEqual(ast.unparse(calls[-1].func), "require_passed_inventory")
        self.assertEqual(ast.unparse(calls[-2].func), "dump")


if __name__ == "__main__":
    unittest.main()
