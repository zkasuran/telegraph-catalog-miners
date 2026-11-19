# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Failure controls for portable input integrity and path scanning."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tools
import verify


class PortableInputs(unittest.TestCase):
    def setUp(self):
        (tools.ROOT / 'scratch').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=tools.ROOT / 'scratch')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_binary_and_text_path_controls(self):
        target = self.root / 'test.wasm'
        target.write_bytes(b'\x00asm\x01\x00\x00\x00')
        with patch.object(tools, 'ROOT', self.root):
            self.assertEqual(tools.scan_public_paths()['files_scanned'], 1)
            for marker in [b'/' + b'home/operator/file', b'.' + b'hq/packet', b'/' + b'Users/operator/file']:
                target.write_bytes(b'\x00asm' + marker)
                with self.assertRaisesRegex(ValueError, 'private path marker'):
                    tools.scan_public_paths()

    def test_public_symlink_refused(self):
        target = self.root / 'file'
        target.write_bytes(b'plain')
        (self.root / 'link').symlink_to(target)
        with patch.object(tools, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                tools.scan_public_paths()

    def test_unrelated_environment_not_inherited(self):
        with patch.object(tools, 'ROOT', self.root), patch.dict('os.environ', {'UNRELATED_SECRET': 'dummy', 'GOFLAGS': '-mod=mod'}):
            environment = tools.clean_env()
        self.assertNotIn('UNRELATED_SECRET', environment)
        self.assertEqual(environment['GOFLAGS'], '-mod=vendor')
        self.assertEqual(environment['GOPROXY'], 'off')
        self.assertEqual(environment['GOTOOLCHAIN'], 'local')
        self.assertEqual(environment['GOMODCACHE'], str(self.root / 'scratch/go-mod'))

    def test_wrong_executable_refused_before_execution(self):
        executable = self.root / 'wrong-rustc'
        executable.write_bytes(b'not the pinned compiler')
        executable.chmod(0o700)
        (self.root / 'toolchain.json').write_text(json.dumps({'rustc': {'sha256': '0' * 64, 'version_marker': 'not reached'}}))
        with patch.object(tools, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                tools.Tools(SimpleNamespace(rustc=str(executable)))

    def test_changed_or_extra_validation_input_refused(self):
        kit = self.root / 'validation'
        kit.mkdir()
        source = kit / 'reference.go'
        source.write_bytes(b'package comparator\n')
        (kit / 'PINS.json').write_text(json.dumps({'kit_commit': verify.KIT_PIN,
            'files': {'validation/reference.go': {'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}))
        with patch.object(verify, 'ROOT', self.root), patch.object(verify, 'KIT', kit), patch.object(tools, 'ROOT', self.root):
            self.assertEqual(len(verify.validate_bundle()['files']), 1)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'input drift'):
                verify.validate_bundle()
            source.write_bytes(b'package comparator\n')
            (kit / 'extra.go').write_bytes(b'package comparator\n')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                verify.validate_bundle()


if __name__ == '__main__':
    unittest.main()
