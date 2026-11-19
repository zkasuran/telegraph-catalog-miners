# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only controls. Synthetic wire fixtures are not model inference."""
import json
import os
import struct
import subprocess
import sys
import unittest
from pathlib import Path

from contracts import FOUNDATION, literal_response, metrics, quality, read_split, score_bits, source_manifest
from preserve import stamp
from proxy import identity, input_root, validate_request
from serve import config


class Contracts(unittest.TestCase):
    def setUp(self):
        self.row = {"keccak":"1"*64}
        batch = [{"tokens":[50281,22,50282]}]
        self.request = {"batch":batch,**identity(self.row),"input_root":input_root(batch)}
        self.response = {k:v for k,v in self.request.items() if k != "batch"}
        self.response["results"] = [{"score":1065353216}]  # Synthetic 1.0 bits.

    def test_literal_bare_wire_and_alias_refusal(self):
        self.assertEqual(literal_response(self.response,self.request),[1065353216])
        for key in ("foundation_hash","adapter_hash","tokenizer_hash"):
            for value in ("0x"+self.response[key],self.response[key].upper()+" "):
                wrong = {**self.response,key:value}
                with self.assertRaises(ValueError):
                    literal_response(wrong,self.request)
                wrong_request = {**self.request,key:value}
                with self.assertRaises(ValueError):
                    validate_request(wrong_request,self.row)

    def test_fp32_nonfinite_and_wrong_result_count(self):
        for value in (True,-1,1 << 32,0x7f800000,0xff800000,0x7fc00000):
            with self.assertRaises(ValueError):
                score_bits(value)
        for results in ([],[{},{}],[{"error":"not a score"}]):
            with self.assertRaises(ValueError):
                literal_response({**self.response,"results":results},self.request)

    def test_frozen_identity_and_source_manifest(self):
        value = source_manifest()
        self.assertEqual(value["original_groups"],32)
        self.assertEqual(value["groups_by_split"],{"train":16,"validation":8,"audit":8})
        self.assertEqual(len(read_split("train")),128)
        self.assertEqual(len(read_split("audit")),64)

    def test_quality_gate_does_not_relax(self):
        trained = {"accuracy":.80,"per_label":{"true":.70,"false":.90}}
        self.assertTrue(quality(trained,{"accuracy":.75},.50))
        self.assertFalse(quality({**trained,"accuracy":.79},{"accuracy":.50},.50))
        self.assertFalse(quality({**trained,"per_label":{"true":.69,"false":.91}},{"accuracy":.50},.50))
        self.assertFalse(quality(trained,{"accuracy":.76},.50))
        self.assertFalse(quality(trained,{"accuracy":.50},.71))

    def test_constructed_stamp_preserves_tensor_bytes(self):
        # This is a self-authored zero tensor parser fixture, not model weights.
        header = json.dumps({"fixture":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode()
        raw = struct.pack("<Q",len(header))+header+bytes(4)
        out = stamp(raw,{"sand":"test licence fixture"})
        size = struct.unpack("<Q",out[:8])[0]
        self.assertEqual(out[8+size:],bytes(4))
        self.assertEqual(json.loads(out[8:8+size])["fixture"],json.loads(header)["fixture"])
        self.assertIn("test licence fixture",out.decode(errors="replace"))

    def test_both_remote_paths_refuse_local_operation(self):
        for name in ("remote_train.py","serve.py"):
            result = subprocess.run([sys.executable,str(Path(__file__).parent/name)],capture_output=True,text=True,
                                    timeout=10,env={"PATH":os.defpath,"PYTHONDONTWRITEBYTECODE":"1"})
            self.assertNotEqual(result.returncode,0)
            self.assertIn("model operations require the controller",result.stderr)

    def test_serving_rejects_unbound_or_cross_namespace_inputs(self):
        with self.assertRaises(ValueError):
            config({})


if __name__ == "__main__":
    unittest.main()
