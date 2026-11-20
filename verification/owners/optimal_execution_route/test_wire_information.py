# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from wire_information import MAX_ROWS, MAX_TEXT_BYTES, assess, diagnostic, render


def row(group="a", q="constructed question", a="A", b="B", label=1):
    return {"source_group": group * 64, "question": q, "answer_a": a, "answer_b": b, "preferred": label}


class WireInformationTests(unittest.TestCase):
    def test_retained_primary_diagnostic_is_not_new_model_data(self):
        got = diagnostic()
        self.assertEqual(got["optimistic_wire_correct_ceiling"], {"numerator": 1, "denominator": 2})
        self.assertEqual(got["duplicate_rows_removed"], 1)
        self.assertEqual(got["same_byte_length_pairs"], 2)
        self.assertEqual(got["unavoidable_wire_errors"], 1)
        self.assertFalse(got["registration_ready"])
        self.assertEqual(got["model_operations"], 0)

    def test_literal_renderer_keeps_whitespace_and_case(self):
        self.assertEqual(render(" q ", " A "), "### Task:\n q \n\n### Answer:\n A ")
        self.assertNotEqual(render("q", "A"), render("Q", "A"))

    def test_diagnostic_requires_unchanged_source_binding(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="route-wire-source-") as directory:
            root = Path(directory)
            for name in ("source-manifest.json", "policy.json", "evidence/STATE.json", "evidence/STATE-OLDER.json", "primary/render_fragment.py.txt"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, path)
            fragment = root / "primary/render_fragment.py.txt"
            fragment.write_bytes(fragment.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                diagnostic(root)
            shutil.copyfile(source / "primary/render_fragment.py.txt", fragment)
            manifest = json.loads((root / "source-manifest.json").read_text())
            manifest["records"] = [r for r in manifest["records"] if r["path"] != "primary/render_fragment.py.txt"]
            (root / "source-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "absent"):
                diagnostic(root)

    def test_conflicting_labels_and_reversed_candidates(self):
        got = assess([row(), row("b", a="B", b="A", label=1)])
        self.assertEqual(got["conflicting_wire_pairs"], 1)
        self.assertEqual(got["unavoidable_wire_errors"], 1)
        self.assertEqual(got["optimistic_answer_only_correct_ceiling"], {"numerator": 1, "denominator": 2})

    def test_duplicate_reversal_does_not_create_independence(self):
        got = assess([row(), row(a="B", b="A", label=-1)])
        self.assertEqual(got["duplicate_rows_removed"], 1)
        self.assertEqual(got["declared_source_groups"], 1)
        self.assertFalse(got["source_groups_independently_verified"])

    def test_supplied_context_changes_wire_but_not_answer_only(self):
        got = assess([row(q="constructed state A"), row("b", q="constructed state B", label=-1)])
        self.assertEqual(got["unavoidable_wire_errors"], 0)
        self.assertEqual(got["optimistic_answer_only_correct_ceiling"], {"numerator": 1, "denominator": 2})

    def test_identical_answers_cannot_satisfy_strict_preference(self):
        got = assess([row(a="A", b="A"), row("b", a="A", b="A", label=0)])
        self.assertEqual(got["optimistic_wire_correct_ceiling"], {"numerator": 1, "denominator": 2})

    def test_ties_are_a_distinct_label(self):
        got = assess([row(label=0), row("b", label=0), row("c", label=1)])
        self.assertEqual(got["optimistic_wire_correct_ceiling"], {"numerator": 2, "denominator": 3})

    def test_fail_closed_schema_type_and_group(self):
        for field, value in (("preferred", True), ("preferred", 2), ("source_group", "A" * 64), ("answer_a", 1)):
            bad = row()
            bad[field] = value
            with self.assertRaises(ValueError):
                assess([bad])
        bad = row()
        bad["reference_state"] = 123
        with self.assertRaises(ValueError):
            assess([bad])

    def test_ceiling_and_surrogate_controls(self):
        for rows in ([], [row()] * (MAX_ROWS + 1), [row(a="X" * (MAX_TEXT_BYTES + 1))], [row(a="\ud800")]):
            with self.assertRaises(ValueError):
                assess(rows)
        with self.assertRaises(ValueError):
            assess([row(a="X" * MAX_TEXT_BYTES)] * 130)

    def test_no_normalized_standin_merges_distinct_wire(self):
        got = assess([row(), row("b", a="A ", label=-1)])
        self.assertEqual(got["wire_pair_groups"], 2)
        self.assertEqual(got["unavoidable_wire_errors"], 0)

    def test_optimistic_bound_matches_exhaustive_pair_choices(self):
        rows = [row(), row("b", label=-1), row("c", a="C", b="D", label=0)]
        best = max(sum(choice[0 if r["answer_a"] == "A" else 1] == r["preferred"] for r in rows)
                   for choice in itertools.product((-1, 0, 1), repeat=2))
        self.assertEqual(assess(rows)["optimistic_wire_correct_ceiling"]["numerator"], best)
        self.assertEqual(assess(copy.deepcopy(rows)), assess(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
