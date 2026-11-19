# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import unittest

from contracts import FAMILY
from replay import rewritten_header
from training_record import validate_record


class ReplayTests(unittest.TestCase):
    def fixture(self):
        return {"__metadata__": {"training": json.dumps({"family": FAMILY, "trained": True, "audit_required": True}),
                                 "x-sand": "original terms", "x-source-notice": "original CC0 notice", "intent": FAMILY},
                "head.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}

    def test_recovery_only_changes_training_metadata(self):
        old = self.fixture()
        frozen = copy.deepcopy(old)
        new = rewritten_header(old, True)
        self.assertEqual(old, frozen)
        self.assertEqual(new["head.bias"], old["head.bias"])
        self.assertEqual(new["__metadata__"]["x-sand"], old["__metadata__"]["x-sand"])
        self.assertEqual(new["__metadata__"]["x-original-training-provenance"], old["__metadata__"]["training"])
        record = json.loads(new["__metadata__"]["training"])
        validate_record(record)
        self.assertEqual(record["children"][0]["accuracy"], 0.5)
        self.assertFalse(record["children"][0]["held_out"])

    def test_initializer_keeps_provenance_without_false_training_record(self):
        old = self.fixture()
        old["__metadata__"]["training"] = json.dumps({"family": FAMILY, "trained": False})
        new = rewritten_header(old, False)
        self.assertNotIn("training", new["__metadata__"])
        self.assertEqual(json.loads(new["__metadata__"]["x-original-training-provenance"])["trained"], False)


if __name__ == "__main__":
    unittest.main()
