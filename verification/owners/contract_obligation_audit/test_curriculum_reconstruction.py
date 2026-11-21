# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored provenance mutations, not evaluation rows or model performance."""
import copy
import unittest

from contracts import FAMILY, FORMATS, sha
from hans_curriculum import LABELS, REVISION, TRAIN_SHA256, require_curriculum, require_reconstruction


class CurriculumReconstructionTests(unittest.TestCase):
    rebuild_check = staticmethod(require_reconstruction)

    def setUp(self):
        self.expected = {"schema_version": 1, "source_revision": REVISION, "source_sha256": TRAIN_SHA256,
            "original_training_rows_read": 30000, "selected_originals": 320, "selected_views": 640,
            "original_labels": {label: 160 for label in LABELS}, "heuristics": {"authored-heuristic": 320},
            "shortcuts": {style: {key: .5 for key in ("length", "unigram_overlap", "answer_only_bow", "exact_quote", "hypothesis_prior")} for style in FORMATS},
            "coefficients": {"authored-integer-coefficient": 0}, "evaluation_rows_read": False,
            "synthetic_auxiliary_training_only": True, "human_preference_reviews": False, "new_audited_contract_claim": False,
            "selected_pair_ids": ["authored-source-" + str(i) for i in range(320)],
            "selected_normalized_premise_sha256": [sha(("authored-premise-" + str(i)).encode()) for i in range(320)],
            "selected_originals_sha256": sha(b"authored-originals"), "constructed_views_sha256": sha(b"authored-views"),
            "licence": "MIT, copyright 2019 R. Thomas McCoy; retain original notice"}
        self.manifest = {**copy.deepcopy(self.expected), "family": FAMILY, "run_id": 1, "commit": "a" * 40, "source_manifest_sha256": "b" * 64}
        self.freeze = {"syntactic_curriculum_originals": 320, "family": FAMILY, "run_id": 1, "commit": "a" * 40,
                       "source_manifest_sha256": "b" * 64, "syntactic_curriculum_manifest_sha256": "c" * 64}

    def check(self):
        # Every mutation passes the original shape/hash gate with a coherent
        # authored freeze, then must fail independent source reconstruction.
        require_curriculum(self.freeze, self.manifest, "c" * 64)
        self.rebuild_check(self.manifest, self.expected)

    def test_primary_exact_reconstruction_accepts(self):
        self.check()

    def test_wrong_original_pair_id_refuses(self):
        self.manifest["selected_pair_ids"][0] = "authored-wrong-source-record"
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_normalized_premise_refuses(self):
        self.manifest["selected_normalized_premise_sha256"][0] = sha(b"authored-wrong-premise")
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_original_record_fingerprint_refuses(self):
        self.manifest["selected_originals_sha256"] = sha(b"authored-wrong-records")
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_literal_preference_views_refuses(self):
        self.manifest["constructed_views_sha256"] = sha(b"authored-wrong-views")
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_integer_coefficients_refuses(self):
        self.manifest["coefficients"] = {"authored-integer-coefficient": 100}
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
