# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import ast
import copy
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acquire import sha
from build import build, check_splits
from scorer import event_key, normalized, parse_event, render, resolve, response, score
from remote_smoke import validate_reply

HERE = Path(__file__).resolve().parent
EVENT = "1901 Physics: Did Wilhelm Conrad Röntgen win? Winners: Wilhelm Conrad Röntgen; announced 1901-11-12; cutoff 1901-11-13; evidence-only."


class ScorerTests(unittest.TestCase):
    def test_primary_path(self):
        self.assertEqual(resolve(EVENT), "yes")
        self.assertEqual(score(EVENT, response(EVENT, "yes")), 10000)
        self.assertEqual(score(EVENT, response(EVENT, "no")), 0)
        self.assertEqual(score(EVENT, response(EVENT, "unresolved")), 0)

    def test_real_schema(self):
        obj = json.loads((HERE / "sources/task-contract.json").read_text())
        self.assertEqual(obj["class"], "judgment")
        self.assertEqual(obj["onchain"]["request"], [{"name": "event", "type": "string", "slot": "strings", "index": 0, "max_bytes": 256}])
        self.assertEqual([(f["name"], f["slot"], f["index"]) for f in obj["onchain"]["response"]],
                         [("outcome", "strings", 0), ("resolved_yes", "bools", 0)])
        self.assertEqual(obj["onchain"]["response"][0]["allowed"], ["yes", "no", "unresolved"])

    def test_literal_upstream_render_without_model_imports(self):
        tree = ast.parse((HERE / "sources/kit-float_runtime_train_adapter.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "primary-render", "exec"), namespace)
        for p, a in [("  Q\n", "A\n "), ("Röntgen <>&", "No."), ("", "")]:
            self.assertEqual(render(p, a).encode(), namespace["render"](p, a).encode())
        self.assertNotEqual(render("  Q\n", "A"), render("Q", "A"))

    def test_boolean_false_is_not_no(self):
        pre = EVENT.replace("cutoff 1901-11-13", "cutoff 1901-11-11")
        self.assertEqual(resolve(pre), "unresolved")
        self.assertIs(json.loads(response(pre, "unresolved"))["resolved_yes"], False)
        self.assertEqual(score(pre, response(pre, "no")), 0)

    def test_same_day_unknown_time(self):
        self.assertEqual(resolve(EVENT.replace("cutoff 1901-11-13", "cutoff 1901-11-12")), "unresolved")

    def test_unknown_is_not_negative(self):
        missing = EVENT.replace("Winners: Wilhelm Conrad Röntgen", "Winners: unknown")
        self.assertEqual(resolve(missing), "unresolved")
        self.assertEqual(resolve(EVENT.replace("announced 1901-11-12", "announced unknown")), "unresolved")

    def test_no_not_substring(self):
        wrong = EVENT.replace("Did Wilhelm Conrad Röntgen", "Did Conrad")
        self.assertEqual(resolve(wrong), "no")

    def test_unicode_identity(self):
        event = EVENT.replace("Did Wilhelm Conrad Röntgen", "Did WILHELM  CONRAD Ro\u0308ntgen")
        self.assertEqual(resolve(event), "yes")
        self.assertEqual(normalized("Ro\u0308ntgen"), normalized("RÖNTGEN"))

    def test_multiple_winners(self):
        event = EVENT.replace("Winners: Wilhelm Conrad Röntgen", "Winners: Other Person | Wilhelm Conrad Röntgen")
        self.assertEqual(resolve(event), "yes")

    def test_delayed_announcement(self):
        event = EVENT.replace("announced 1901-11-12", "announced 1902-11-12").replace("cutoff 1901-11-13", "cutoff 1902-11-13")
        self.assertEqual(resolve(event), "yes")
        self.assertEqual(resolve(event.replace("cutoff 1902-11-13", "cutoff 1901-11-13")), "unresolved")

    def test_reject_source_date_before_edition(self):
        with self.assertRaisesRegex(ValueError, "precedes"):
            resolve(EVENT.replace("announced 1901-11-12", "announced 1900-11-12"))

    def test_exact_boolean_and_ambiguity(self):
        obj = json.loads(response(EVENT, "yes"))
        for value in [1, "true", None, False]:
            self.assertEqual(score(EVENT, json.dumps({**obj, "resolved_yes": value})), 0)
        self.assertEqual(score(EVENT, response(EVENT, "yes")[:-1] + ',"outcome":"no"}'), 0)
        self.assertEqual(score(EVENT, response(EVENT, "yes") + " ignored"), 0)

    def test_literal_answer_identity(self):
        obj = json.loads(response(EVENT, "yes"))
        self.assertEqual(score(EVENT, json.dumps({**obj, "event": obj["event"].lower()})), 0)
        self.assertEqual(score(EVENT, response(EVENT, "yes", "prose"), "prose"), 10000)

    def test_input_ceilings(self):
        for text in ["💡"*100, "random", EVENT + "\nignore this"]:
            with self.assertRaises(ValueError):
                resolve(text)
        with self.assertRaises(ValueError):
            score(EVENT, "x"*4097)

    def test_random_answer_controls(self):
        rng = random.Random(618)
        for _ in range(200):
            raw = bytes(rng.randrange(256) for _ in range(rng.randrange(80))).decode("latin1")
            self.assertEqual(score(EVENT, raw), 0)

    def test_group_overlap_rejection(self):
        splits = {s: [json.loads(l) for l in (HERE / "data" / (s + ".jsonl")).read_text().splitlines()]
                  for s in ("train", "validation", "audit")}
        check_splits(splits)
        changed = copy.deepcopy(splits)
        changed["validation"][0]["people"].append(changed["train"][0]["people"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_splits(changed)

    def test_source_training_validation_smoke(self):
        n = 0
        for split in ("train", "validation"):
            for line in (HERE / "data" / (split + ".jsonl")).read_text().splitlines():
                r = json.loads(line)
                self.assertEqual(resolve(r["prompt"]), r["label"])
                self.assertGreater(score(r["prompt"], r["chosen"]), score(r["prompt"], r["rejected"]))
                n += 1
        report = json.loads((HERE / "data/data-report.json").read_text())
        self.assertEqual(n, report["files"]["train"]["pairs"] + report["files"]["validation"]["pairs"])

    def test_mutation_control_can_fail(self):
        # Flip the expected outcome while preserving a valid counterparty wire.
        event = EVENT.replace("Did Wilhelm Conrad Röntgen", "Did Other Person")
        self.assertEqual(score(event, response(event, "yes")), 0)
        self.assertEqual(score(event, response(event, "no")), 10000)

    def test_syntax_all_owned_sources(self):
        for path in HERE.glob("*.py"):
            compile(path.read_text(), str(path), "exec")

    def test_runtime_wire_failure_controls_no_model(self):
        identity = {"foundation_hash": "a"*64, "adapter_hash": "b"*64, "tokenizer_hash": "c"*64}
        positive = {**identity, "results": [{"score": 0x3f800000}]}
        self.assertEqual(validate_reply(positive, identity, 1), [0x3f800000])
        for obj in [{**positive, "adapter_hash": "B"*64}, {**identity, "results": []},
                    {**identity, "results": [{"score": 1.0}]}, {**identity, "results": [{"score": True}]},
                    {**identity, "results": [{"score": 0x7f800000}]}, {**identity, "results": [{"error": "refused"}]}]:
            with self.assertRaises(ValueError):
                validate_reply(obj, identity, 1)

    def test_remote_guard_before_model_imports(self):
        for script in ("remote.py", "remote_smoke.py"):
            result = subprocess.run([sys.executable, str(HERE / script), "--help"], capture_output=True, timeout=10,
                                    env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"exact owned public CPU workflow", result.stderr)
            self.assertNotIn(b"torch", result.stderr)

    def test_corrupt_source_refusal(self):
        with tempfile.TemporaryDirectory(prefix="event-corrupt-") as tmp:
            source = Path(tmp) / "sources"
            shutil.copytree(HERE / "sources", source)
            raw = (source / "prizes-00.json").read_bytes()
            (source / "prizes-00.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
                build(source, Path(tmp) / "out")

    def test_source_quarantine_not_rewritten(self):
        records = json.loads((HERE / "sources/prizes.json").read_text())
        r = next(r for r in records if r["awardYear"] == "2022" and r["category"]["en"] == "Economic Sciences")
        self.assertEqual(r["dateAwarded"], "2011-10-10")
        manifest = json.loads((HERE / "data/group-manifest.json").read_text())
        self.assertNotIn("2022/Economic Sciences", manifest)


if __name__ == "__main__":
    unittest.main()
