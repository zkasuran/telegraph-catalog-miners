# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Attach retained upstream grants to successful and failed artifacts."""
import json
from pathlib import Path
import shutil
import struct
import zipfile
from source_gate import HERE, digest

SHARED = HERE


def terms(source):
    files = {"own-sand": SHARED / "LICENSE", "upstream-apache-2.0": SHARED / "UPSTREAM-APACHE-2.0.txt",
             "upstream-mit": SHARED / "KIT-LICENSE", "upstream-cc-by-4.0": SHARED / "CC-BY-4.0.html"}
    result = {"x-" + k: p.read_text() for k, p in files.items()}
    if digest(files["own-sand"].read_bytes()) != "e6cf0cc83fbc51616590209d71ef1fa5fe46c130bef44e0d97d439ee91663447":
        raise ValueError("reviewed SAND terms mismatch")
    if digest(files["upstream-apache-2.0"].read_bytes()) != "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4":
        raise ValueError("retained Apache terms mismatch")
    if digest(files["upstream-mit"].read_bytes()) != "e20635bc37e1ec73cc962c20cf23b0043bfcf20d01dbcefb959238ebc2f8fde1":
        raise ValueError("retained primary MIT terms mismatch")
    if "reproduce and Share" not in result["x-upstream-cc-by-4.0"]:
        raise ValueError("missing source CC BY grant")
    result["x-notice"] = (HERE / "NOTICE").read_text()
    result["x-licence"] = "LicenseRef-zkasuran-SAND-1.0"
    return result


def stamp(path, metadata):
    if path.is_symlink() or path.stat().st_size > 64 << 20:
        raise ValueError("artifact ceiling")
    if path.suffix == ".safetensors":
        raw = path.read_bytes()
        if len(raw) < 8:
            raise ValueError("truncated tensor container")
        n = struct.unpack("<Q", raw[:8])[0]
        if not 0 < n <= 1 << 20 or 8 + n > len(raw):
            raise ValueError("tensor header ceiling")
        header = json.loads(raw[8:8+n])
        if not isinstance(header, dict) or not isinstance(header.get("__metadata__", {}), dict):
            raise ValueError("tensor header object")
        if all(header.get("__metadata__", {}).get(k) == v for k, v in metadata.items()):
            return
        header["__metadata__"] = {**header.get("__metadata__", {}), **metadata}
        encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        if len(encoded) > 1 << 20:
            raise ValueError("stamped header ceiling")
        output = struct.pack("<Q", len(encoded)) + encoded + raw[8+n:]
        if output[8+len(encoded):] != raw[8+n:]:
            raise ValueError("tensor payload changed")
        path.write_bytes(output)
    elif path.suffix == ".pt":
        with zipfile.ZipFile(path, "a") as archive:
            names = archive.namelist()
            roots = [n.removesuffix("data.pkl") for n in names if n.endswith("/data.pkl")]
            if len(roots) != 1 or len(names) > 256:
                raise ValueError("unsupported Torch ZIP layout")
            member = roots[0] + "llm-output-licences.json"
            encoded = json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode()
            if member in names:
                if archive.read(member) != encoded:
                    raise ValueError("checkpoint licence mismatch")
            else:
                archive.writestr(member, encoded)
    else:
        raise ValueError("unsupported artifact")


def preserve(root):
    source = root / "source"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SHARED / "LICENSE", root / "LICENSE")
    shutil.copyfile(HERE / "NOTICE", root / "NOTICE")
    if not (source / "CC-BY-4.0.html").exists():
        return
    metadata = terms(source)
    for name in ("UPSTREAM-APACHE-2.0.txt", "KIT-LICENSE"):
        shutil.copyfile(SHARED / name, root / name)
    shutil.copyfile(source / "CC-BY-4.0.html", root / "CC-BY-4.0.html")
    for path in sorted(root.rglob("*.safetensors")) + sorted(root.rglob("*.pt")):
        # Never change immutable downloaded foundation or base artifacts.
        if "base" not in path.parts and "foundation" not in path.parts and "telegraph-wasm-baseline" not in str(path):
            stamp(path, metadata)
