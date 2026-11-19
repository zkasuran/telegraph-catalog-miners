# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import copy
import json
import os
import random
import subprocess
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare import check_splits, fit_counts, render
from scorer import resolve, score
from source import make_prompt, pairs, sentence_candidates, sha

HERE = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.rows = [json.loads(line) for line in (HERE / "data/train.jsonl").read_text().splitlines()]
        self.row = self.rows[0]

    def test_primary_path_real_original(self):
        row = self.row
        self.assertEqual(score(row["prompt"], row["chosen"]), 1)
        self.assertEqual(score(row["prompt"], row["rejected"]), 0)
        original = (HERE / "evidence/xml" / row["source_path"]).read_bytes()
        candidates, disposition = sentence_candidates(original, row["source_path"], row["git_blob_sha1"])
        self.assertEqual(disposition, "eligible")
        self.assertTrue(any(c["evidence"] == row["evidence"] for c in candidates))
        self.assertEqual(sha(original), row["source_sha256"])

    def test_changed_claim_flips_verdict_same_source(self):
        row = self.row
        mirror = pairs(row, row["group_id"])
        self.assertEqual({resolve(r["prompt"])[1] for r in mirror}, {"Yes", "No"})
        for r in mirror:
            self.assertEqual(score(r["prompt"], r["chosen"]), 1)
            self.assertEqual(score(r["prompt"], r["rejected"]), 0)

    def test_changed_evidence_changes_decision(self):
        row = self.row
        changed = make_prompt(row["opposite_claim"], row["claim"])
        self.assertEqual(resolve(changed)[1], "No")

    def test_no_invisible_evidence(self):
        row = self.row
        for prompt in (row["claim"], make_prompt("", row["claim"]), make_prompt("[citation only]", row["claim"]),
                       make_prompt(row["evidence"], "Unrelated entity increased.")):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                resolve(prompt)

    def test_literal_wire_not_normalized_standin(self):
        row = self.row
        for answer in (row["chosen"].lower(), " " + row["chosen"], json.dumps({"answer": row["chosen"]}),
                       row["chosen"] + " Source count: 99."):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                score(row["prompt"], answer)
        with self.assertRaises(ValueError):
            resolve(row["prompt"].replace("\n", "\r\n"))
        self.assertEqual(render(" <x>\n", "a & b"), "### Task:\n <x>\n\n\n### Answer:\na & b")

    def test_source_hash_entities_and_overflow_fail(self):
        row = self.row
        raw = (HERE / "evidence/xml" / row["source_path"]).read_bytes()
        for altered in (raw + b" ", b'<!ENTITY injected "x">' + raw, b"x" * ((4 << 20) + 1)):
            with self.subTest(size=len(altered)), self.assertRaises(ValueError):
                sentence_candidates(altered, row["source_path"], row["git_blob_sha1"])
        with self.assertRaises(ValueError):
            resolve("x" * 257)

    def test_split_overlap_control_can_fail(self):
        splits = {name: [json.loads(line) for line in (HERE / "data" / (name + ".jsonl")).read_text().splitlines()]
                  for name in ("train", "validation", "audit")}
        self.assertEqual(check_splits(splits)["component"], 0)
        bad = copy.deepcopy(splits)
        bad["validation"][0]["source_doi"] = bad["train"][0]["source_doi"]
        with self.assertRaises(ValueError):
            check_splits(bad)
        bad = copy.deepcopy(splits)
        bad["validation"][0]["evidence"] = "  " + bad["train"][0]["evidence"].upper() + "  "
        with self.assertRaises(ValueError):
            check_splits(bad)

    def test_immutable_empirical_coefficients(self):
        frozen = json.loads((HERE / "data/frozen-controls.json").read_text())
        calculated = fit_counts(self.rows)
        self.assertEqual(frozen["answer_only_coefficients"], calculated["answer_only_coefficients"])
        self.assertEqual(frozen["surface_coefficients"], calculated["surface_coefficients"])
        self.assertTrue(all(w == 0 for w in frozen["answer_only_coefficients"].values()))
        self.assertEqual(frozen["surface_coefficients"], [0] * 10)
        self.assertEqual(frozen["fit_split"], "train")

    def test_random_parser_inputs_refused(self):
        rng = random.Random(42)
        for _ in range(256):
            prompt = "".join(chr(rng.randrange(32, 127)) for _ in range(rng.randrange(0, 300)))
            with self.assertRaises(ValueError):
                resolve(prompt)

    def test_actual_cli_honest_and_tampered(self):
        row = self.row
        cmd = [sys.executable, "-W", "error", str(HERE / "scorer.py")]
        honest = subprocess.run(cmd, input=json.dumps({"question": row["prompt"], "answer": row["chosen"]}).encode(),
                                capture_output=True, timeout=5)
        self.assertEqual(honest.returncode, 0, honest.stderr)
        self.assertEqual(json.loads(honest.stdout)["score"], 1)
        changed = subprocess.run(cmd, input=json.dumps({"question": row["claim"], "answer": row["chosen"]}).encode(),
                                 capture_output=True, timeout=5)
        self.assertEqual(changed.returncode, 1)

    def test_neural_entrypoint_refuses_local_execution(self):
        cmd = [sys.executable, "-W", "error", str(HERE / "remote.py"), "--root", str(HERE / "must-not-exist")]
        env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
        completed = subprocess.run(cmd, capture_output=True, timeout=5, env=env)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"bounded public CPU", completed.stderr)
        self.assertFalse((HERE / "must-not-exist").exists())

    def test_failed_checkpoint_terms_do_not_change_payload(self):
        from terms import metadata, preserve, stamp
        with tempfile.TemporaryDirectory(prefix="research-terms-") as directory:
            root = Path(directory)
            payload = struct.pack("<f", 1.25)
            header = json.dumps({"toy_test_only": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            path = root / "test-container.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            stamp(path)
            raw = path.read_bytes()
            size = struct.unpack("<Q", raw[:8])[0]
            self.assertEqual(raw[8 + size:], payload)
            terms = json.loads(raw[8:8 + size])["__metadata__"]
            self.assertEqual(len(json.loads(terms["x-source-attribution"])), 112)
            self.assertIn("does not restrict", terms["x-upstream-notice"])
            before = raw
            stamp(path)
            self.assertEqual(path.read_bytes(), before)
            preserve(root)
            self.assertTrue((root / "source-attribution.json").is_file())
            self.assertEqual(metadata()["x-licence"], "LicenseRef-zkasuran-SAND-1.0")

    def test_failed_zip_checkpoint_storage_unchanged(self):
        from terms import stamp
        with tempfile.TemporaryDirectory(prefix="research-zip-") as directory:
            path = Path(directory) / "toy-test-only.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("toy/data.pkl", b"not real model data")
                archive.writestr("toy/data/0", b"literal test payload")
            with zipfile.ZipFile(path) as archive:
                before = {i.filename: (i.header_offset, i.CRC, i.file_size) for i in archive.infolist()}
            stamp(path)
            with zipfile.ZipFile(path) as archive:
                for name, original in before.items():
                    item = archive.getinfo(name)
                    self.assertEqual((item.header_offset, item.CRC, item.file_size), original)
                terms = json.loads(archive.read("toy/qa-licences.json"))
                self.assertEqual(len(json.loads(terms["x-source-attribution"])), 112)


if __name__ == "__main__":
    unittest.main()
