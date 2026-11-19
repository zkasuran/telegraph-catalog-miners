# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source/build regression and failure controls, never local model inference."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import data
import licensing
import remote
import runtime_audit
from source_gate import digest, normal
import wire


def original(task="Return 2 + 2.", output="4", rating=4, gid="0" * 64):
    return {"prompt": task, "response": output, "helpfulness": rating, "group_id": gid,
            "source_file": "train.jsonl.gz", "source_row": 1}


class OwnerTests(unittest.TestCase):
    def test_nested_evaluation_not_inner_answer(self):
        row = data.pair(original())
        self.assertEqual(json.loads(row["prompt"]), {"task": "Return 2 + 2.", "output": "4"})
        self.assertEqual(json.loads(row["chosen"])["rating"], "Excellent")
        self.assertNotEqual(row["chosen"], "4")
        self.assertFalse(row["human_review_of_conversion"])

    def test_all_grades_direction_and_semantics(self):
        for grade in range(5):
            r = data.pair(original(rating=grade))
            a, b = json.loads(r["chosen"]), json.loads(r["rejected"])
            self.assertEqual(a["rating"], data.RATINGS[grade])
            self.assertEqual(b["rating"], data.RATINGS[(grade+2) % 5])
            self.assertEqual(a["assessment"], b["assessment"])
            self.assertNotEqual(a["score"], b["score"])

    def test_source_never_cropped(self):
        for task, output in (("x" * 257, "fine"), ("fine", "é" * 129)):
            with self.assertRaisesRegex(ValueError, "256 bytes"):
                data.pair(original(task, output))
        r = data.pair(original("é" * 128, "x" * 256))
        self.assertEqual(json.loads(r["prompt"])["output"], "x" * 256)

    def test_invalid_labels_refused(self):
        for label in (True, 0.5, -1, 5, "4"):
            with self.assertRaises(ValueError):
                data.pair(original(rating=label))

    def test_connected_transitive_normalized_overlap(self):
        rows = [original("Ａ", "one"), original("a", "two"), original("other", " TWO ")]
        self.assertEqual(len(data.connected_groups(rows)), 1)
        self.assertEqual(normal("Ａ\n Ｂ"), "a b")

    def test_leaking_split_can_fail(self):
        a = data.pair(original())
        b = data.pair(original(gid="1" * 64))
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.validate_splits({"train": [a], "validation": [b], "audit": []})

    def test_exact_rational_baseline_frozen(self):
        train = [data.pair(original(rating=i, gid=str(i) * 64)) for i in range(5)]
        baseline = data.fit_count_baseline(train)
        self.assertEqual(baseline["coefficients"], {})
        self.assertEqual(data.control_scores(train, baseline)["answer_only_counts"], 0.5)
        self.assertEqual(baseline, data.fit_count_baseline(train))

    def test_local_neural_entry_refused_before_import(self):
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "public GitHub CPU"):
            remote.remote_only()

    def test_f32_literal_failure_controls(self):
        self.assertEqual(runtime_audit.decode(0x3f800000), 1.0)
        for value in (True, -1, 0x100000000, 0x7f800000, 0x7fc00000, "1065353216"):
            with self.assertRaises(ValueError):
                runtime_audit.decode(value)

    def test_stamp_preserves_payload_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.safetensors"
            h = json.dumps({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0,4]}}).encode()
            payload = struct.pack("<f", 1.0)
            path.write_bytes(struct.pack("<Q", len(h)) + h + payload)
            licensing.stamp(path, {"x-own-sand": "test terms only"})
            raw = path.read_bytes()
            size = struct.unpack("<Q", raw[:8])[0]
            self.assertEqual(raw[8+size:], payload)
            licensing.stamp(path, {"x-own-sand": "test terms only"})
            self.assertEqual(raw, path.read_bytes())
            path.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "truncated"):
                licensing.stamp(path, {})

    def test_literal_caller_fixture_and_mutation(self):
        # Synthetic controls test the verifier, not proof of the actual caller.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "caller.txt"
            source.write_text("synthetic test source, not node evidence")
            cases = []
            for task in ("Add 2 and 2.", 'Return <&".\n', "Return empty."):
                p = data.prompt(task, "4")
                a = data.evaluation("4", 4)
                cases.append({"task": task, "output": "4", "question_key": p,
                              "raw_response_body": a, "scored_answer": a, "rendered_text": data.render(p, a)})
            obj = {"family": data.FAMILY, "protected_material": False,
                   "caller": {"commit": "0" * 40, "source_file": "caller.txt", "source_sha256": digest(source.read_bytes())},
                   "cases": cases}
            path = root / "fixture.json"
            path.write_text(json.dumps(obj))
            self.assertEqual(wire.verify(path)["literal_cases"], 3)
            obj["cases"][1]["question_key"] = json.dumps(json.loads(cases[1]["question_key"]))
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "literal question_key"):
                wire.verify(path)
            obj["cases"][1] = {**cases[1], "question_key": data.prompt(cases[1]["task"], "4"), "scored_answer": "Excellent"}
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "complete raw"):
                wire.verify(path)

    def test_fixture_hash_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            (path.parent / "caller.txt").write_text("modified")
            path.write_text(json.dumps({"family": data.FAMILY, "protected_material": False,
              "caller": {"commit": "0" * 40, "source_file": "caller.txt", "source_sha256": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                wire.verify(path)


if __name__ == "__main__":
    unittest.main()
