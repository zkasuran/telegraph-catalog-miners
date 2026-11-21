# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored startup cleanup controls. No model process, SSH or forwarding runs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from contracts import FOUNDATION, file_sha
from executor import ReferenceExecutor
from serve_remote import hold


class StartupCleanupTests(unittest.TestCase):
    def root(self):
        temporary = tempfile.TemporaryDirectory(prefix="contract-startup-cleanup-", dir="/tmp/opencode")
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def tracked_open(self):
        opened = []
        original = Path.open
        def tracked(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            opened.append(stream)
            self.addCleanup(stream.close)
            return stream
        return opened, tracked

    def test_engine_launch_refusal_closes_its_log(self):
        root = self.root()
        opened, tracked = self.tracked_open()
        with patch("executor.remote_only"), patch("executor.subprocess.Popen", side_effect=OSError("authored launch refusal")), patch.object(Path, "open", tracked):
            with self.assertRaises(OSError):
                ReferenceExecutor(root, root / "authored-not-a-model.safetensors", "a" * 64)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)
        partial = ReferenceExecutor.__new__(ReferenceExecutor)
        partial.log = (root / "authored-wait-refusal.log").open("w")
        self.addCleanup(partial.log.close)
        partial.process = Mock()
        partial.process.poll.return_value = None
        partial.process.wait.side_effect = OSError("authored process-wait refusal")
        with self.assertRaises(OSError):
            partial.close()
        self.assertTrue(partial.log.closed)

    def test_healthy_mocked_startup_keeps_pinned_environment_and_closes(self):
        root = self.root()
        process = Mock()
        process.poll.return_value = None
        with patch("executor.remote_only"), patch("executor.subprocess.Popen", return_value=process) as popen, patch("executor.socket.create_connection"):
            value = ReferenceExecutor(root, root / "authored-not-a-model.safetensors", "a" * 64)
        self.assertEqual(popen.call_args.kwargs["env"]["FOUNDATION_MODEL_HASH"], FOUNDATION["foundation_lora_fp32.onnx"])
        self.assertFalse(value.log.closed)
        value.close()
        self.assertTrue(value.log.closed)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    def serving_fixture(self):
        root = self.root()
        package = root / "authored-not-a-model.safetensors"
        package.write_bytes(b"authored opaque bytes, no model")
        report = root / "registration-ready.json"
        report.write_text(json.dumps({"runtime": {"vectors": []}}))
        state = {"package": str(package), "report_sha256": file_sha(report),
                 "row": {"sha256": file_sha(package), "keccak": "a" * 64}}
        (root / "serving-state.json").write_text(json.dumps(state))
        return root

    def test_server_bind_refusal_closes_created_backend(self):
        root = self.serving_fixture()
        backend = Mock()
        with patch("serve_remote.remote_only"), patch("serve_remote.ReferenceExecutor", return_value=backend), patch("serve_remote.Server", side_effect=OSError("authored bind refusal")):
            with self.assertRaises(OSError):
                hold(root)
        backend.close.assert_called_once()
        backend.score.assert_not_called()

    def test_thread_start_refusal_closes_server_and_backend_without_forwarding(self):
        root = self.serving_fixture()
        backend, server, thread = Mock(), Mock(), Mock()
        thread.start.side_effect = RuntimeError("authored thread-start refusal")
        with patch("serve_remote.remote_only"), patch("serve_remote.ReferenceExecutor", return_value=backend), patch("serve_remote.Server", return_value=server), patch("serve_remote.threading.Thread", return_value=thread), patch("serve_remote.subprocess.Popen") as forwarding:
            with self.assertRaises(RuntimeError):
                hold(root)
        backend.close.assert_called_once()
        server.server_close.assert_called_once()
        server.shutdown.assert_not_called()
        forwarding.assert_not_called()
        backend.score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
