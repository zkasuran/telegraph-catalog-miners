# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import json
import os
import sys
import tarfile
from pathlib import Path

from fetch_source import fetch
from foundation_pins import FOUNDATION, RELEASE

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / 'runtime-sources.json').read_text())

def pinned_prepare(root):
    model, base = (root / 'foundation', root / 'base')
    model.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, sha in FOUNDATION.items():
        fetch(RELEASE + name, model / name, sha)
    (base / 'tokenizer.json').write_bytes((model / 'tokenizer.json').read_bytes())
    spec = SOURCES['base']
    for name, sha in spec['files'].items():
        fetch(f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}", base / name, sha)
    commit = SOURCES['kit']['revision']
    archive = root / 'kit.tar.gz'
    fetch('https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/' + commit, archive, SOURCES['kit']['sha256'], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError('oversized reference kit')
        source.extractall(root, filter='data')
    kit = root / ('telegraph-wasm-baseline-' + commit)
    sys.path.insert(0, str(kit))
    os.environ['HF_HUB_OFFLINE'] = '1'
    return (model, base, kit)

def prepare_reference(root):
    from runtime_check import model_remote_only
    model_remote_only()
    return pinned_prepare(root)

def prepare_kit(root):
    pin = SOURCES['kit']
    archive = root / 'kit.tar.gz'
    fetch('https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/' + pin['revision'], archive, pin['sha256'], 32 << 20)
    with tarfile.open(archive) as source:
        if len(source.getmembers()) > 5000 or sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError('reference kit expansion ceiling')
        source.extractall(root, filter='data')
    return root / ('telegraph-wasm-baseline-' + pin['revision'])
