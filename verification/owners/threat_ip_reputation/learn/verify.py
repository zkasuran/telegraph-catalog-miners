# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Public learner source gate. Neural imports are blocked locally."""
import ast
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path


class SourceOnly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"torch", "tensorflow", "onnxruntime", "transformers", "peft", "numpy"}:
            raise RuntimeError("neural/fitted-model imports forbidden in source gate")


sys.meta_path.insert(0, SourceOnly())
import contracts
import test_learner
import test_training_record
import test_replay
import test_class_binding
import test_wire
import test_variant2
import test_warmstart
import test_audit_source
import test_history

HERE = Path(__file__).resolve().parent


def main():
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    tree = ast.parse((HERE / "remote.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name.partition(".")[0] in {"torch", "transformers", "peft", "numpy", "tokenizers"} for alias in node.names):
                raise ValueError("remote code contains top-level neural imports")
    data = contracts.read_development(HERE / "data")
    stream = io.StringIO()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(test_learner.LearnerTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_training_record.TrainingRecordTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_replay.ReplayTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_class_binding.LiveClassTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_wire.WireTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_variant2.Variant2Tests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_warmstart.WarmStartTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_audit_source.AuditSourceTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(test_history.HistoryTests)])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise RuntimeError("source learner gate failed")
    print(json.dumps({"status": "SOURCE_LEARNER_GATE_PASSED", "tests": result.testsRun,
                      "train_pairs": len(data["train"]), "validation_pairs": len(data["validation"]),
                      "train_sha256": contracts.TRAIN_SHA, "validation_sha256": contracts.VALIDATION_SHA,
                      "remote_primary_path_refused_locally": True, "model_operations": 0, "qualification_ready": False}))


if __name__ == "__main__":
    main()
