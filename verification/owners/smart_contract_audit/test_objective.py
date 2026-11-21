# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from objective import SETTINGS, load, scalar_loss, softplus


class ObjectiveChecks(unittest.TestCase):
    def test_primary_truth_order_and_offset_penalty(self):
        self.assertAlmostEqual(scalar_loss(0, 0), math.log(2))
        self.assertLess(scalar_loss(2, -2), scalar_loss(-2, 2))
        # Pair ranking alone ignores a common offset. Truth calibration does not.
        self.assertAlmostEqual(softplus(-(2-(-2))), softplus(-(5-1)))
        self.assertGreater(scalar_loss(5, 1), scalar_loss(2, -2))
        self.assertTrue(math.isfinite(scalar_loss(1000, -1000)))

    def test_closed_config_refuses_audit_or_gate_change(self):
        source = Path(__file__).with_name(SETTINGS)
        record, _ = load(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for key, value in (("target_source", "audit"), ("quality_gates_changed", True), ("audit_used_for_selection", True), ("denominator", 0), ("ranking_numerator", True)):
                path.write_text(json.dumps({**record, key: value}))
                with self.assertRaises(ValueError):
                    load(path)

    def test_actual_tensor_objective_refuses_local_execution(self):
        from objective_remote import loss, check
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Model loading"):
                loss(None, None, None)
            with self.assertRaisesRegex(RuntimeError, "Model loading"):
                check("unused")
