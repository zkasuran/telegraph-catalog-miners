# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data


class DataTests(unittest.TestCase):
    def row(self, text, label="Positive", split="train"):
        return {"text": text, "label": label, "split": split, "source_id": text, "domain": "test"}

    def test_normalization_reconciles_case_spacing_and_unicode(self):
        self.assertEqual(data.group_id("Ａ Good  Review"), data.group_id("a good review"))

    def test_conflicting_groups_are_removed_not_moved_into_test(self):
        rows = [self.row(f"source text {split} {label}", label, split)
                for split in data.LIMITS for label in ("Positive", "Negative")]
        rows += [self.row("same Original"), self.row("SAME original", split="test")]
        with patch.dict(data.LIMITS, {"train": 2, "validation": 2, "test": 2}):
            splits, removed = data.select(rows)
        self.assertEqual(removed, 2)
        self.assertFalse(any(r["group_id"] == data.group_id("same original")
                             for group in splits.values() for r in group))

    def test_actual_overlap_is_refused(self):
        row = data.pair("SENTIMENT_ANALYSIS", {**self.row("an original text"), "group_id": data.group_id("an original text")})
        other = data.pair("SENTIMENT_ANALYSIS", {**self.row("a different text", "Negative"), "group_id": data.group_id("a different text")})
        with self.assertRaisesRegex(ValueError, "overlapping"):
            data.check_splits({"train": [row, other], "test": [row, other]})

    def test_an_invented_group_id_cannot_hide_overlap(self):
        row = data.pair("SENTIMENT_ANALYSIS", {**self.row("an original text"), "group_id": "fake-distinct-id"})
        with self.assertRaisesRegex(ValueError, "does not match"):
            data.check_splits({"train": [row]})

    def test_answers_change_only_the_assessment_not_the_source_quote(self):
        row = {**self.row("It works as promised."), "group_id": "ab"}
        p = data.pair("SENTIMENT_ANALYSIS", row, 0)
        good, bad = json.loads(p["chosen"]), json.loads(p["rejected"])
        self.assertEqual(good["reason"], bad["reason"])
        self.assertEqual(good["sentiment"], "Positive")
        self.assertEqual(bad["sentiment"], "Negative")
        self.assertEqual(len(p["chosen"]), len(p["rejected"]))

    def test_moderation_verdict_and_bool_stay_consistent(self):
        row = {**self.row("public comment", "Flagged"), "group_id": "ab"}
        p = data.pair("CONTENT_MODERATION", row, 0)
        self.assertTrue(json.loads(p["chosen"])["flagged"])
        self.assertFalse(json.loads(p["rejected"])["flagged"])

    def test_download_hash_failure_never_leaves_accepted_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "input"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"tampered")), self.assertRaisesRegex(ValueError, "hash mismatch"):
                data.fetch("https://example.com/input", path, "0" * 64)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".partial").exists())

    def test_unicode_line_separators_are_content_not_jsonl_records(self):
        row = {"prompt": "first\u0085second\u2028third", "chosen": "yes", "rejected": "no"}
        raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        self.assertEqual(data.decode_jsonl(raw), [row])

    def test_shortcut_direction_is_frozen_not_selected_on_test(self):
        train = [{"prompt": "original text", "chosen": "long good answer", "rejected": "short"}]
        test = [{"prompt": "other text", "chosen": "short", "rejected": "long bad answer"}]
        result = data.shortcut_baselines({"train": train, "test": test})
        self.assertEqual(result["by_split"]["train"]["length"], 1.0)
        self.assertEqual(result["by_split"]["test"]["length"], 0.0)


if __name__ == "__main__":
    unittest.main()
