# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Human-source policy, grouped holdout and prompt-conditioned failure controls."""
import json
import unittest
from collections import Counter

from aegis import contrast, grouped, group_id, select, single, source_spec, training_pairs


class AegisTests(unittest.TestCase):
    def row(self, i, label):
        text = "Public synthetic source fixture " + str(i)
        return {"text": text, "label": label, "group_id": group_id(text), "source_id": str(i)}

    def test_current_complete_inventory_and_original_grant(self):
        value = source_spec()
        self.assertEqual(len(value["files"]), 5)
        self.assertEqual(value["upstream_license"], "CC-BY-4.0")

    def test_human_annotation_redaction_and_late_cross_split_conflicts(self):
        rows = [("train.json", "train", {"id": 1, "prompt": "A public sample message", "prompt_label": "safe", "prompt_label_source": "human", "response_label_source": "llm_jury"}),
                ("test.json", "test", {"id": 2, "prompt": "a  public sample message", "prompt_label": "safe", "prompt_label_source": "human"}),
                ("train.json", "train", {"id": 3, "prompt": "Another public sample message", "prompt_label": "unsafe", "prompt_label_source": "llm_jury"}),
                ("train.json", "train", {"id": 4, "prompt": "REDACTED", "prompt_label": "unsafe", "prompt_label_source": "human", "reconstruction_id_if_redacted": "private reconstruction ID"})]
        clean, counts = grouped(rows, set())
        self.assertEqual(clean, [])
        self.assertEqual(counts["scanned_rows"], 4)
        self.assertEqual(counts["cross_split_or_annotation_conflict"], 1)

    def test_actual_conditional_pairs_force_prompt_and_preserve_human_source_ids(self):
        a, b = self.row(1, "Safe"), self.row(2, "Flagged")
        first, second = contrast(a, b, 0, 0), contrast(a, b, 0, 1)
        self.assertEqual(first["chosen"], second["rejected"])
        self.assertEqual(first["rejected"], second["chosen"])
        self.assertNotEqual(first["prompt"], second["prompt"])
        self.assertEqual(first["source_group_ids"], [a["group_id"], b["group_id"]])
        self.assertFalse(json.loads(first["chosen"])["flagged"])
        self.assertTrue(json.loads(second["chosen"])["flagged"])

    def test_whole_training_epoch_is_balanced_unique_and_not_stock_negative_padding(self):
        rows = [self.row(i, "Safe") for i in range(792)] + [self.row(i + 1000, "Flagged") for i in range(792)]
        for epoch in (0, 1):
            pairs = training_pairs(rows, epoch)
            self.assertEqual(len(pairs), 1584)
            self.assertEqual(Counter(row["label"] for row in pairs), {"Safe": 792, "Flagged": 792})
            negatives = Counter(row["rejected"] for row in pairs)
            self.assertGreaterEqual(len(negatives) / len(pairs), 0.80)
            self.assertLessEqual(max(negatives.values()) / len(pairs), 0.02)
            self.assertEqual(Counter(row["style"] for row in pairs), {0: 528, 1: 528, 2: 528})
            lengths = sum(len(r["chosen"]) > len(r["rejected"]) for r in pairs) / len(pairs)
            self.assertEqual(lengths, 0.5)

    def test_missing_source_inventory_insufficient_groups_and_exclusions_refuse(self):
        rows, _ = grouped([("train.json", "train", {"id": 1, "prompt": "A public unique message", "prompt_label": "safe", "prompt_label_source": "human"})], {group_id("A public unique message")})
        self.assertEqual(rows, [])
        with self.assertRaises(ValueError):
            select(rows, {"train": 2}, {"train": "frozen-source-salt"}, lambda row: True)
        with self.assertRaises(ValueError):
            training_pairs([], 0)

    def test_literal_schema_matches_processed_enum_and_boolean_contract(self):
        row = self.row(1, "Flagged")
        pair = single(row, 0)
        self.assertEqual(json.loads(pair["prompt"]), {"text": row["text"]})
        self.assertEqual(json.loads(pair["chosen"]), {"verdict": "Flagged", "flagged": True})
        self.assertEqual(json.loads(pair["rejected"]), {"verdict": "Safe", "flagged": False})

    def test_all_remote_entry_points_refuse_before_neural_imports(self):
        import sys
        from pathlib import Path
        import aegis_remote
        import aegis_export
        for operation in (lambda: aegis_remote.prepare(Path("not-created")), lambda: aegis_remote.fit(Path("not-created")),
                          lambda: aegis_remote.audit(Path("not-created")), lambda: aegis_export.export(Path("not-created"), "unused", {})):
            with self.assertRaises(RuntimeError):
                operation()
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "peft", "tokenizers")))

    def test_frozen_source_model_context_audit_and_terms_parse_without_models(self):
        import hashlib
        import struct
        from pathlib import Path
        root = Path(__file__).resolve().parent
        plan = json.loads((root / "aegis-plan.json").read_bytes())
        self.assertEqual((plan["epochs"], plan["optimizer_calls"], plan["max_tokens"]), (2, 198, 512))
        self.assertEqual((plan["training_sources"], plan["validation_sources"], plan["audit_sources"]), (1584, 256, 256))
        self.assertEqual(len(set(json.loads((root / "aegis-exclusions.json").read_bytes()))), 768)
        for name, expected in ((plan["warm_start"]["checkpoint"], plan["warm_start"]["checkpoint_sha256"]),
                               (plan["warm_start"]["untrained"], plan["warm_start"]["untrained_sha256"])):
            raw = (root / "assets" / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            size = struct.unpack("<Q", raw[:8])[0]
            metadata = json.loads(raw[8:8 + size])["__metadata__"]
            for term in ("notice", "apache_2_0", "kit_mit", "sand"):
                self.assertTrue(metadata["x-terms-" + term])
        self.assertIn("produce, reproduce, and Share Adapted Material", (root / "CC-BY-4.0.txt").read_text())
