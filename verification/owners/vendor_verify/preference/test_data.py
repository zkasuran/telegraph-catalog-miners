# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Artificial mechanics controls, not new human preferences or task truth."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import baselines
import data


def source(i=0, strength=-1):
    return {
        "split": "train",
        "prompt": f"Artificial instruction {i}",
        "response_1": f"Artificial first response {i}",
        "response_2": f"Artificial second response {i}",
        "preference_strength": strength,
        "three_most_similar_preferences": [{"strength": strength}] * 3,
    }


def pair(i=0):
    result, _ = data.direct_preference(source(i))
    return {**result, "source_row_sha256": data.sha(data.encode(source(i)))}


class Contracts(unittest.TestCase):
    def test_primary_literal_render(self):
        self.assertEqual(
            data.render("P\nQ", "R\nS"), "### Task:\nP\nQ\n\n### Answer:\nR\nS"
        )

    def test_source_signs(self):
        self.assertEqual(
            data.direct_preference(source())[0]["source_preferred_response"], 1
        )
        self.assertEqual(
            data.direct_preference(source(strength=1))[0]["source_preferred_response"],
            2,
        )

    def test_validation_skipped_before_any_text(self):
        for split in ("val", "validation"):
            self.assertEqual(
                data.direct_preference({"split": split}),
                (None, "official_validation_excluded"),
            )
        with self.assertRaises(ValueError):
            data.direct_preference({"split": "test"})

    def test_invalid_neutral_and_disagreement(self):
        for value in (0, -100, True, float("nan"), float("inf"), 4, "1"):
            self.assertIsNone(data.direct_preference(source(strength=value))[0])
        row = source()
        row["three_most_similar_preferences"] = [{"strength": 1}] * 3
        self.assertIsNone(data.direct_preference(row)[0])
        row["three_most_similar_preferences"] = [
            {"strength": -3},
            {"strength": 0},
            {"strength": -1},
        ]
        self.assertIsNone(data.direct_preference(row)[0])

    def test_normalized_identical_refused(self):
        row = source()
        row["response_1"], row["response_2"] = "Ａ B", "a  b"
        self.assertEqual(
            data.direct_preference(row)[1], "normalized_identical_answers_excluded"
        )

    def test_source_full_scan_no_first_bad_row_stop(self):
        rows = [source(0), {"split": "val"}, source(1), source(2, 0), source(3)]
        raw = gzip.compress(b"".join(data.encode(row) for row in rows), mtime=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.gz"
            path.write_bytes(raw)
            pairs, counts = data.read_source(path, data.sha(raw), len(raw))
            self.assertEqual(len(pairs), 3)
            self.assertEqual(counts["records"], 5)
            self.assertEqual(counts["official_validation_excluded"], 1)
            with self.assertRaises(ValueError):
                data.read_source(path, "0" * 64, len(raw))

    def test_group_union_precedes_filter_and_keeps_originals(self):
        a, b, c = pair(0), pair(1), pair(2)
        b["chosen"] = a["chosen"]
        c["prompt"] = b["prompt"]
        pool, _ = data.grouped([a, b, c])
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["component_originals"], 3)
        self.assertTrue(
            any(
                all(pool[0][k] == original[k] for k in data.FIELDS)
                for original in (a, b, c)
            )
        )

    def test_conflicting_preferences_remove_whole_group(self):
        a = pair(0)
        b = {**a, "chosen": a["rejected"], "rejected": a["chosen"]}
        pool, removed = data.grouped([a, b])
        self.assertEqual(pool, [])
        self.assertEqual(removed["conflicting_components"], 1)

    def test_allocation_determinism_and_no_padding(self):
        rows = [pair(i) for i in range(9)]
        pool, _ = data.grouped(rows)
        reverse, _ = data.grouped(list(reversed(rows)))
        self.assertEqual(pool, reverse)
        splits = data.allocate(pool, {"train": 3, "validation": 3, "audit": 3})
        self.assertTrue(data.check_splits(splits))
        with self.assertRaises(ValueError):
            data.allocate(pool)

    def test_overlap_detection_in_both_directions(self):
        pool, _ = data.grouped([pair(i) for i in range(9)])
        splits = data.allocate(pool, {"train": 3, "validation": 3, "audit": 3})
        for field in (*data.FIELDS, "group_id"):
            broken = copy.deepcopy(splits)
            broken["audit"][0][field] = broken["train"][0][field]
            with self.assertRaises(ValueError):
                data.check_splits(broken)

    def test_frozen_baseline_tamper_and_no_audit_fit(self):
        raw = b"".join(data.encode(pair(i)) for i in range(4))
        frozen = baselines.fit_training(raw)
        self.assertEqual(
            baselines.load_frozen(data.encode(frozen), data.sha(raw)), frozen
        )
        with self.assertRaises(ValueError):
            baselines.load_frozen(data.encode(frozen), "0" * 64)
        broken = copy.deepcopy(frozen)
        broken["idf_hex"]["artificial"] = "inf"
        with self.assertRaises(ValueError):
            baselines.load_frozen(data.encode(broken), data.sha(raw))
        with tempfile.TemporaryDirectory() as tmp:
            train, audit, output = [
                Path(tmp) / n for n in ("train.jsonl", "audit.jsonl", "BASELINES.json")
            ]
            train.write_bytes(raw)
            audit.write_bytes(raw)
            with self.assertRaises(ValueError):
                baselines.freeze_once(audit, output)
            baselines.freeze_once(train, output)
            with self.assertRaises(ValueError):
                baselines.freeze_once(train, output)

    def test_symlink_and_existing_experiment_refused_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                data.bounded_read(link, 16)
            with self.assertRaises(ValueError):
                data.prepare(root / "missing", root / "missing", root / "missing", root)

    def test_no_neural_modules_imported(self):
        import sys

        self.assertFalse({"torch", "transformers", "peft"} & set(sys.modules))

    def test_invalid_json_source_fails(self):
        raw = gzip.compress(b"not json\n", mtime=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.gz"
            path.write_bytes(raw)
            with self.assertRaises(json.JSONDecodeError):
                data.read_source(path, data.sha(raw), len(raw))


if __name__ == "__main__":
    unittest.main()
