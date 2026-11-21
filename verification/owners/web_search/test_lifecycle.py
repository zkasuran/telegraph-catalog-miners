# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real lifecycle functions with synthetic subprocesses. No neural work or keys."""
import json
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import serve
from common import FOUNDATION, HOST
from contracts import sha

REAL_POPEN = subprocess.Popen


class ServingLifecycle(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path("/synthetic-owner-runtime")
        self.raw = b"harmless lifecycle fixture, not a model package"
        self.row = {"sha256": sha(self.raw), "keccak": "a" * 64, "forward_port": 48601,
                    "runnable_url": "https://tga1.zkasuran.dev/_s2o_web_search_aaaaaaaa/runnable"}
        self.executor, self.forward = MagicMock(), MagicMock()
        for process in (self.executor, self.forward):
            process.poll.return_value = None
            process.wait.return_value = 0
        self.popen = self.stack.enter_context(patch.object(serve.subprocess, "Popen", side_effect=[self.executor, self.forward]))
        self.stack.enter_context(patch.object(serve, "remote_only"))
        self.stack.enter_context(patch.dict(serve.os.environ, {"PATH": "/synthetic-bin"}, clear=True))
        self.unlink = self.stack.enter_context(patch.object(Path, "unlink", autospec=True))

        def bytes_for(path):
            if path == self.root / "serving-state.json":
                return json.dumps(self.row).encode()
            if path == self.root / "WEB_SEARCH.safetensors":
                return self.raw
            raise AssertionError("attempted to open another file, including a key")
        self.stack.enter_context(patch.object(Path, "read_bytes", autospec=True, side_effect=bytes_for))
        self.socket = self.stack.enter_context(patch.object(serve.socket, "socket"))
        self.probe = self.socket.return_value.__enter__.return_value
        self.probe.getsockname.return_value = ("127.0.0.1", 43210)
        self.stack.enter_context(patch.object(serve.socket, "create_connection"))
        self.backend = self.stack.enter_context(patch.object(serve, "Backend"))
        self.server = self.stack.enter_context(patch.object(serve, "BoundedServer"))
        self.proxy = self.server.return_value
        self.proxy.server_port = 43211
        self.proxy.fatal.is_set.return_value = False
        self.thread_factory = self.stack.enter_context(patch.object(serve.threading, "Thread"))
        self.thread = self.thread_factory.return_value
        # One ready probe and one forwarding iteration, then bounded expiry.
        self.clock = self.stack.enter_context(patch.object(serve.time, "monotonic", side_effect=[0, 0, 0, 0, 10**9]))
        self.stack.enter_context(patch.object(serve.time, "sleep"))

    def assert_keys_released(self):
        # No private file is created, read or exposed. Only unlink calls observed.
        self.assertEqual(self.unlink.call_args_list, [call(self.root / "forward-key", missing_ok=True),
                                                     call(self.root / "forward-key.pub", missing_ok=True)])

    def test_primary_serving_path_preserves_wire_and_stops_owned_processes(self):
        serve.serve(self.root)
        command, = self.popen.call_args_list[0].args
        self.assertEqual(command, [str(self.root / "adapterexec"), "--package", str(self.root / "WEB_SEARCH.safetensors"), "--listen", "127.0.0.1:43210"])
        env = self.popen.call_args_list[0].kwargs["env"]
        self.assertEqual(env["FOUNDATION_MODEL_HASH"], FOUNDATION["foundation_lora_fp32.onnx"])
        self.assertEqual(env["FOUNDATION_ENGINE_WASM_HASH"], FOUNDATION["foundation_engine.wasm"])
        self.assertEqual(self.server.call_args.args[1], self.row)
        forward_command = self.popen.call_args_list[1].args[0]
        self.assertIn("IdentityAgent=none", forward_command)
        self.assertIn("StrictHostKeyChecking=yes", forward_command)
        self.assertIn("127.0.0.1:48601:127.0.0.1:43211", forward_command)
        self.assertEqual(forward_command[-1], "telegraph-verify@" + HOST)
        self.proxy.shutdown.assert_called_once()
        self.proxy.server_close.assert_called_once()
        self.thread.join.assert_called_once_with(timeout=5)
        for process in (self.forward, self.executor):
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=10)
        self.assert_keys_released()

    def test_executor_startup_exception_releases_keys_without_starting_forward(self):
        self.popen.side_effect = OSError("synthetic executor startup failure")
        with self.assertRaisesRegex(OSError, "startup failure"):
            serve.serve(self.root)
        self.assertEqual(self.popen.call_count, 1)
        self.server.assert_not_called()
        self.assert_keys_released()

    def test_package_mismatch_refuses_startup_and_releases_keys(self):
        self.row["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "package bytes changed"):
            serve.serve(self.root)
        self.popen.assert_not_called()
        self.assert_keys_released()

    def test_socket_startup_failure_releases_keys(self):
        self.probe.bind.side_effect = OSError("synthetic bind failure")
        with self.assertRaisesRegex(OSError, "bind failure"):
            serve.serve(self.root)
        self.popen.assert_not_called()
        self.assert_keys_released()

    def test_exited_executor_is_reaped_without_termination(self):
        self.executor.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "reference engine failed"):
            serve.serve(self.root)
        self.executor.terminate.assert_not_called()
        self.executor.wait.assert_called_once_with(timeout=5)
        self.assert_keys_released()

    def test_proxy_construction_failure_stops_executor_and_releases_keys(self):
        self.server.side_effect = OSError("synthetic proxy failure")
        with self.assertRaisesRegex(OSError, "proxy failure"):
            serve.serve(self.root)
        self.executor.terminate.assert_called_once()
        self.assert_keys_released()

    def test_unstarted_thread_does_not_shutdown_or_join_but_closes_socket(self):
        self.thread.start.side_effect = RuntimeError("synthetic thread startup failure")
        with self.assertRaisesRegex(RuntimeError, "thread startup failure"):
            serve.serve(self.root)
        self.proxy.shutdown.assert_not_called()
        self.thread.join.assert_not_called()
        self.proxy.server_close.assert_called_once()
        self.executor.terminate.assert_called_once()
        self.assert_keys_released()

    def test_proxy_cleanup_error_does_not_skip_process_or_key_cleanup(self):
        self.proxy.shutdown.side_effect = OSError("synthetic proxy shutdown failure")
        with self.assertRaisesRegex(OSError, "proxy shutdown failure"):
            serve.serve(self.root)
        self.proxy.server_close.assert_called_once()
        self.thread.join.assert_called_once_with(timeout=5)
        for process in (self.forward, self.executor):
            process.terminate.assert_called_once()
        self.assert_keys_released()

    def test_forward_cleanup_error_does_not_skip_executor_or_keys(self):
        self.forward.terminate.side_effect = OSError("synthetic forward cleanup failure")
        with self.assertRaisesRegex(OSError, "forward cleanup failure"):
            serve.serve(self.root)
        self.executor.terminate.assert_called_once()
        self.proxy.server_close.assert_called_once()
        self.assert_keys_released()

    def test_bounded_kill_fallback_for_each_owned_process(self):
        for process in (self.forward, self.executor):
            process.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 10), 0]
        serve.serve(self.root)
        for process in (self.forward, self.executor):
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_args_list, [call(timeout=10), call(timeout=5)])
        self.assert_keys_released()

    def test_one_key_cleanup_failure_still_releases_the_other(self):
        self.unlink.side_effect = [OSError("synthetic unlink failure"), None]
        with self.assertRaisesRegex(OSError, "unlink failure"):
            serve.serve(self.root)
        self.assert_keys_released()
        self.executor.terminate.assert_called_once()

    def test_stop_process_reaps_a_real_harmless_child(self):
        # A Python sleeper, not an executor, neural import, key or network call.
        process = REAL_POPEN([sys.executable, "-c", "import time; time.sleep(30)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            serve.stop_process(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    def test_remote_guard_refuses_before_any_resource_access(self):
        from remote import remote_only
        with patch.object(serve, "remote_only", remote_only):
            with self.assertRaisesRegex(RuntimeError, "bounded public CPU workflow"):
                serve.serve(self.root)
        self.popen.assert_not_called()
        self.unlink.assert_not_called()


if __name__ == "__main__":
    unittest.main()
