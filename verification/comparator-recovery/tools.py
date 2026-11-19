#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Explicit pinned executables and isolated environments for the release gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name in ['rustc', 'cargo', 'go', 'cast', 'node', 'ruff']:
        parser.add_argument('--' + name, default=os.environ.get('ZKCMP_' + name.upper(), name),
                            help='Pinned executable on PATH or an explicit executable path')
    return parser.parse_args()


def clean_env():
    scratch = ROOT / 'scratch'
    for name in ['home', 'tmp', 'cargo', 'go-cache', 'go-path', 'go-mod', 'xdg']:
        (scratch / name).mkdir(parents=True, exist_ok=True)
    return {'PATH': os.environ.get('PATH', os.defpath), 'HOME': str(scratch / 'home'),
            'TMPDIR': str(scratch / 'tmp'), 'CARGO_HOME': str(scratch / 'cargo'),
            'RUSTFLAGS': '-Dwarnings', 'CARGO_NET_OFFLINE': 'true',
            'PYTHONDONTWRITEBYTECODE': '1', 'LANG': 'C.UTF-8',
            'GOTOOLCHAIN': 'local', 'GOENV': 'off', 'GOWORK': 'off',
            'GOCACHE': str(scratch / 'go-cache'), 'GOPATH': str(scratch / 'go-path'),
            'GOMODCACHE': str(scratch / 'go-mod'), 'GOTMPDIR': str(scratch / 'tmp'),
            'GOPROXY': 'off', 'GOSUMDB': 'off', 'GOTELEMETRY': 'off',
            'GOFLAGS': '-mod=vendor', 'CGO_ENABLED': '0', 'GOMAXPROCS': '4',
            'XDG_CONFIG_HOME': str(scratch / 'xdg'), 'XDG_CACHE_HOME': str(scratch / 'xdg')}


class Tools:
    def __init__(self, args):
        pins = json.loads((ROOT / 'toolchain.json').read_bytes())
        self.paths = {}
        self.proof = {}
        self.environment = clean_env()
        for name in pins:
            found = shutil.which(str(getattr(args, name)))
            if not found:
                raise ValueError('executable not found: ' + name)
            path = Path(found).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != pins[name]['sha256']:
                raise ValueError('executable hash mismatch: ' + name + '; select the pinned executable directly')
            version_args = ['--version', '--verbose'] if name == 'rustc' else ['version'] if name == 'go' else ['--version']
            process = subprocess.run([str(path), *version_args], env=self.environment, check=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
            version = process.stdout.decode()
            if pins[name]['version_marker'] not in version:
                raise ValueError('executable version mismatch: ' + name)
            self.paths[name] = str(path)
            self.proof[name] = {'sha256': digest, 'version': version}
        rust_version = self.proof['rustc']['version']
        if 'commit-hash: 8bab26f4f68e0e26f0bb7960be334d5b520ea452\n' not in rust_version:
            raise ValueError('Rust commit mismatch')
        process = subprocess.run([self.paths['rustc'], '--print', 'sysroot'], env=self.environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True)
        self.sysroot = Path(process.stdout.decode().strip()).resolve()
        self.environment['RUSTC'] = self.paths['rustc']
        self.environment['PATH'] = os.pathsep.join(dict.fromkeys(str(Path(p).parent) for p in self.paths.values())) + os.pathsep + os.defpath

    def executable(self, name):
        return self.paths[name]

    def env(self):
        return dict(self.environment)


def public_files():
    files = []
    for path in ROOT.rglob('*'):
        if any(p in {'scratch', 'evidence', '__pycache__'} for p in path.relative_to(ROOT).parts):
            continue
        if path.is_symlink():
            raise ValueError('public symlink is not allowed')
        if path.is_file():
            files.append(path)
    return sorted(files)


def scan_public_paths():
    markers = [b'/'+b'home/', b'/'+b'Users/', b'/'+b'root/', b'.'+b'hq']
    hits = []
    files = public_files()
    for path in files:
        if path.is_symlink():
            raise ValueError('public symlink is not allowed')
        data = path.read_bytes()
        name = str(path.relative_to(ROOT)).encode()
        if any(marker in data or marker in name for marker in markers):
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise ValueError('private path marker found: ' + ', '.join(hits))
    return {'files_scanned': len(files), 'private_path_hits': 0, 'binary_files_included': True}
