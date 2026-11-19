# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Meaningful pure-data boundary and licence retention tests. No model work."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

from checkpoint_terms import decode_attribution, family_bundle, remote_guard, stamp
from general_sources import Union
from pilot import COUNTS, checks, render_pairs, validated_corpus
from source_probe import sha

HERE = Path(__file__).resolve().parent


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((HERE / "pilot-manifest.json").read_text())

    def test_missing_external_cache_is_refused_without_database_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "missing cache"
            with self.assertRaisesRegex(ValueError, "existing complete validated corpus"):
                validated_corpus(cache)
            self.assertFalse(cache.exists())

    def test_external_cache_reports_must_match_frozen_pins(self):
        spec = json.loads((HERE / "general-sources.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for name in ("general.sqlite", "links.tar.bz2", *spec["files"]):
                (cache / name).touch()
            (cache / "general-index.json").write_text(json.dumps({"source_spec_sha256": "0" * 64, "missing_detailed_identities": 0}))
            (cache / "general-graph.json").write_text(json.dumps({"general_spec_sha256": "0" * 64, "links_sha256": "0" * 64, "missing_reciprocals": 0}))
            with self.assertRaisesRegex(ValueError, "do not match the frozen source pins"):
                validated_corpus(cache)
            self.assertEqual((cache / "general.sqlite").stat().st_size, 0)

    def test_fixed_groups_and_bidirectional_original_target_roles(self):
        for split, count in COUNTS.items():
            rows = render_pairs(self.manifest, split)
            self.assertEqual(len(rows), 4 * count)
            for block in self.manifest["splits"][split]:
                selected = [r for r in rows if r["group_id"] == block["id"]]
                self.assertEqual(len(selected), 4)
                self.assertEqual(sorted(r["chosen_id"] for r in selected), sorted(r["donor_id"] for r in selected))
                self.assertEqual([r["direction"] for r in selected].count("eng-fra"), 2)
                for row in selected:
                    self.assertEqual(row["chosen"], self.manifest["originals"][str(row["chosen_id"])]["text"])
                    self.assertEqual(row["rejected"], self.manifest["originals"][str(row["donor_id"])]["text"])

    def test_all_language_union_and_normalized_alias_identity(self):
        union = Union(10)
        union.join(1, 8)
        union.join(8, 2)
        union.join(3, 4)
        self.assertEqual(union.root(1), union.root(2))
        self.assertNotEqual(union.root(1), union.root(3))
        union.join(2, 3)
        self.assertEqual(union.root(1), union.root(4))

    def test_cross_split_component_leak_is_refused(self):
        bad = copy.deepcopy(self.manifest)
        bad["splits"]["validation"][0]["component_roots"][0] = bad["splits"]["train"][0]["component_roots"][0]
        splits = {s: render_pairs(bad, s) for s in COUNTS}
        # Synthetic token IDs keep this test focused on the split boundary.
        class SourceOnlyTokenizer:
            def encode(self, text):
                from types import SimpleNamespace
                return SimpleNamespace(ids=[1, hash(text), 2])
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            checks(bad, splits, SourceOnlyTokenizer(), lambda p, a: p + a)

    def test_source_refs_and_licences_are_not_cc0(self):
        for sid, row in self.manifest["originals"].items():
            self.assertEqual(row["license"], "CC-BY-2.0-FR")
            self.assertTrue(row["credited_contributor"])
            self.assertTrue(row["sentence_url"].endswith("/" + sid))
            self.assertFalse(row["original_text_changed"])

    def test_failed_source_audit_is_bound_to_exact_frozen_originals(self):
        from archive_failed import ARCHIVE, FAILED_SHA
        review = json.loads((ARCHIVE / "semantic-review.json").read_text())
        old = json.loads((ARCHIVE / "pilot-manifest.json").read_text())
        self.assertEqual(review["status"], "FAIL")
        self.assertFalse(review["workflow_ready"])
        self.assertEqual(review["pilot_manifest_sha256"], FAILED_SHA)
        self.assertEqual(sha(ARCHIVE / "pilot-manifest.json"), FAILED_SHA)
        self.assertEqual(len(review["known_failures"]), 5)
        for issue in review["known_failures"]:
            block = old["splits"][issue["split"]][issue["block_index"]]
            self.assertIn([issue["english_id"], issue["french_id"]], block["links"])
            self.assertEqual(issue["english_evidence"], old["originals"][str(issue["english_id"])]["text"])
            self.assertEqual(issue["french_evidence"], old["originals"][str(issue["french_id"])]["text"])

    def test_source_review_precedes_new_freeze_and_excludes_all_old_components(self):
        pool = json.loads((HERE / "reviewed-source-pool.json").read_text())
        review = json.loads((HERE / "source-pool-review.json").read_text())
        self.assertTrue(self.manifest["source_review_before_split"])
        self.assertEqual(self.manifest["reviewed_pool_sha256"], sha(HERE / "reviewed-source-pool.json"))
        self.assertEqual(pool["review_sha256"], sha(HERE / "source-pool-review.json"))
        self.assertEqual(review["reviewed_blocks"], 400)
        self.assertEqual(len(pool["blocks"]), 236)
        self.assertFalse(review["human_review_claimed"])
        self.assertEqual(len(self.manifest["excluded_failed_components"]), 448)
        banned = set(self.manifest["excluded_failed_components"])
        for blocks in self.manifest["splits"].values():
            for block in blocks:
                self.assertFalse(banned.intersection(block["component_roots"]))
                self.assertFalse(block["bilingual_review"]["human_review_claimed"])

    def test_full_attribution_and_terms_survive_synthetic_container_stamp(self):
        rights = json.loads((HERE / "rights.json").read_text())
        bundle = family_bundle(rights, self.manifest)
        # A synthetic empty container tests metadata retention only. No model,
        # checkpoint or real tensor payload is created or opened locally.
        header = b'{"__metadata__":{}}'
        raw = struct.pack("<Q", len(header)) + header
        output = stamp(raw, bundle)
        size = struct.unpack("<Q", output[:8])[0]
        parsed = json.loads(output[8:8 + size])["__metadata__"]
        self.assertLessEqual(size, 1 << 20)
        restored = decode_attribution(parsed["x-terms-tatoeba-sentence-attribution"])
        self.assertEqual(restored, self.manifest)
        self.assertEqual(json.loads(parsed["x-terms-tatoeba-source-rights"]), rights)
        self.assertEqual(output[8 + size:], b"")

    def test_corrupt_attribution_or_decompression_size_is_refused(self):
        rights = json.loads((HERE / "rights.json").read_text())
        envelope = json.loads(family_bundle(rights, self.manifest)["tatoeba-sentence-attribution"])
        envelope["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            decode_attribution(json.dumps(envelope))
        envelope["uncompressed_bytes"] = 3 << 20
        with self.assertRaises(ValueError):
            decode_attribution(json.dumps(envelope))

    def test_missing_author_or_share_alike_source_refused(self):
        bad = copy.deepcopy(self.manifest)
        bad["originals"][next(iter(bad["originals"]))]["credited_contributor"] = ""
        with self.assertRaises(ValueError):
            family_bundle({}, bad)
        bad = copy.deepcopy(self.manifest)
        bad["source_license"] = "CC-BY-SA-4.0"
        with self.assertRaises(ValueError):
            family_bundle({}, bad)

    def test_publication_is_guarded_while_semantic_audit_failed(self):
        with self.assertRaises(RuntimeError):
            remote_guard()


if __name__ == "__main__":
    unittest.main()
