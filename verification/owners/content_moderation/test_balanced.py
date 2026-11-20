# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic source controls, not new neural training/test data or audit measures."""
import copy
import json
import os
import sys
import unittest
from collections import Counter
from unittest.mock import patch

from balanced import audit_select, check_audit_identity, plan, style_positions, training_batch, training_views, validate_plan
from baselines import WORDS, scores
from contracts import sha
from remote import support


class BalancedTests(unittest.TestCase):
    def rows(self, per_label=384):
        data = support()["core"].load_shared()[0]
        return [data.pair("CONTENT_MODERATION", {"text": "Synthetic source control number " + str(i) + " with " + label,
                "label": label, "group_id": sha((label + str(i)).encode()), "source_id": "synthetic", "domain": "synthetic"})
                for label in ("Flagged", "Safe") for i in range(per_label)]

    def test_actual_source_style_rotation_balances_every_epoch_and_group(self):
        rows = self.rows()
        data = support()["core"].load_shared()[0]
        positions = style_positions(rows)
        seen = {r["group_id"]: set() for r in rows}
        for epoch in range(3):
            changed = training_batch(rows, positions, epoch, data.pair)
            self.assertEqual(Counter((r["label"], r["style"]) for r in changed), {(label, style): 128 for label in ("Flagged", "Safe") for style in range(3)})
            self.assertEqual([r["group_id"] for r in changed], [r["group_id"] for r in rows])
            for row in changed:
                seen[row["group_id"]].add(row["style"])
        self.assertTrue(all(styles == {0, 1, 2} for styles in seen.values()))

    def test_stock_negative_gate_is_not_bypassed_or_called_bare_training(self):
        rows = self.rows()
        data = support()["core"].load_shared()[0]
        bare = [data.pair("CONTENT_MODERATION", {**r, "text": json.loads(r["prompt"])["text"]}, 2) for r in rows]
        self.assertLess(len({r["rejected"] for r in bare}) / len(bare), 0.80)
        self.assertGreater(max(Counter(r["rejected"] for r in bare).values()) / len(bare), 0.02)
        views = training_views(rows, data.pair)
        self.assertEqual(set(views), {"json", "prose", "label_led_source"})
        for values in views.values():
            self.assertGreaterEqual(len({r["rejected"] for r in values}) / len(values), 0.80)
            self.assertLessEqual(max(Counter(r["rejected"] for r in values).values()) / len(values), 0.02)
            for row in values:
                text = json.loads(row["prompt"])["text"]
                self.assertIn(text, row["chosen"])
                self.assertIn(text, row["rejected"])
        self.assertTrue(all(r["chosen"].startswith(r["label"] + ", not ") for r in views["label_led_source"]))

    def test_substantive_verdict_roles_carry_truth_not_bag_or_length_tells(self):
        rows = self.rows(6)
        data = support()["core"].load_shared()[0]
        views = training_views(rows, data.pair)
        coefficients = {"idf": {"safe": 1.0, "flagged": 1.0}, "answer_only": {"weights": {"safe": 0.2, "flagged": -0.8, "verdict": 0.3}},
                        "directions": {name: 1 for name in ("length", "tfidf_overlap", "alarm_keywords", "answer_only")}}
        for name, values in views.items():
            for row in values:
                self.assertNotEqual(row["chosen"], row["rejected"])
                self.assertEqual(len(row["chosen"]), len(row["rejected"]))
                self.assertEqual(Counter(WORDS.findall(row["chosen"].casefold())), Counter(WORDS.findall(row["rejected"].casefold())))
                if name == "json":
                    chosen, rejected = json.loads(row["chosen"]), json.loads(row["rejected"])
                    self.assertEqual(chosen["verdict"], row["label"])
                    self.assertNotEqual(rejected["verdict"], row["label"])
                    self.assertEqual(chosen["excluded_verdict"], rejected["verdict"])
                    self.assertEqual(chosen["flagged"], row["label"] == "Flagged")
                else:
                    self.assertLess(row["chosen"].index(row["label"]), row["chosen"].index({"Safe": "Flagged", "Flagged": "Safe"}[row["label"]]))
            self.assertEqual(scores(values, coefficients), {name: 0.5 for name in coefficients["directions"]})

    def test_train_only_membership_extra_epoch_and_duplicate_controls(self):
        rows = self.rows(6)
        positions = style_positions(rows)
        data = support()["core"].load_shared()[0]
        with self.assertRaises(ValueError):
            training_batch(rows, positions, 3, data.pair)
        with self.assertRaises(ValueError):
            training_batch([{**rows[0], "group_id": "0" * 64}], positions, 0, data.pair)
        with self.assertRaises(ValueError):
            style_positions([*rows, rows[0]])
        with self.assertRaises(ValueError):
            style_positions(rows[:-1])

    def test_full_scan_excludes_exposed_and_conflicting_groups_independent_of_order(self):
        rows = [{"group_id": sha((label + str(i)).encode()), "label": label} for label in ("Flagged", "Safe") for i in range(140)]
        excluded = {rows[0]["group_id"], rows[-1]["group_id"]}
        conflict = rows[4]["group_id"]
        rows.append({"group_id": conflict, "label": "Safe"})
        first, inventory = audit_select(rows, excluded, "synthetic-audit", 128)
        second, _ = audit_select(list(reversed(rows)), excluded, "synthetic-audit", 128)
        self.assertEqual(first, second)
        self.assertEqual(inventory["scanned_rows"], 281)
        self.assertEqual(inventory["conflicting_groups"], 1)
        self.assertEqual(len(first), 256)
        self.assertFalse({r["group_id"] for r in first} & (excluded | {conflict}))
        with self.assertRaises(ValueError):
            audit_select(rows[:20], set(), "synthetic-audit", 128)

    def test_frozen_plan_count_and_source_identity_controls(self):
        value = plan()
        validate_plan(value)
        for key in ("epochs", "source_groups", "audit_groups", "audit_per_label", "job_minutes"):
            changed = copy.deepcopy(value)
            changed[key] = 0
            with self.assertRaises(ValueError):
                validate_plan(changed)

    def test_same_unscored_audit_byte_group_and_count_controls(self):
        raw = b"Synthetic audit identity control, not source rows"
        groups = [sha(str(i).encode()) for i in range(256)]
        value = {"frozen_audit_sha256": sha(raw), "frozen_audit_groups_sha256": sha(json.dumps(groups, separators=(",", ":")).encode())}
        check_audit_identity(raw, groups, value)
        for changed_raw, changed_groups in ((raw + b"x", groups), (raw, groups[:-1]), (raw, [groups[0], *groups[:-1]]), (raw, list(reversed(groups)))):
            with self.assertRaises(ValueError):
                check_audit_identity(changed_raw, changed_groups, value)

    def test_remote_work_refuses_locally_before_any_framework_import(self):
        import balanced_remote
        with patch.dict(os.environ, {}, clear=True):
            for call in (balanced_remote.prepare, balanced_remote.initial, balanced_remote.select, balanced_remote.qualify):
                with self.assertRaises(RuntimeError):
                    call(None)
        for name in ("torch", "numpy", "transformers", "tokenizers", "peft"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
