# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Byte-preserving licence stamping for successful and failed remote deltas."""
import json
import shutil
import struct
import zipfile

from source import HERE, sha

LEGAL = {"LICENSE": "e6cf0cc83fbc51616590209d71ef1fa5fe46c130bef44e0d97d439ee91663447",
         "UPSTREAM-APACHE-2.0.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
         "KIT-LICENSE": "e20635bc37e1ec73cc962c20cf23b0043bfcf20d01dbcefb959238ebc2f8fde1"}
MAX_BYTES = 64 << 20
MAX_HEADER = 1 << 20


def licence_metadata():
    result = {}
    for name, expected in LEGAL.items():
        raw = (HERE / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("reviewed licence bytes mismatch: " + name)
        result[name] = raw.decode()
    return {"x-upstream-apache-2.0": result["UPSTREAM-APACHE-2.0.txt"], "x-upstream-mit": result["KIT-LICENSE"],
            "x-upstream-notice": (HERE / "NOTICE").read_text(), "x-own-sand": result["LICENSE"],
            "x-source-grants": "RELISH annotation originals and OpenAlex metadata retain CC0. Linked paper text excluded.",
            "x-licence": "LicenseRef-zkasuran-SAND-1.0"}


def tensor_parts(raw):
    if not isinstance(raw, bytes) or not 8 <= len(raw) <= MAX_BYTES:
        raise ValueError("bounded safetensors bytes required")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 0 < size <= MAX_HEADER or size + 8 > len(raw):
        raise ValueError("truncated or excessive safetensors header")
    header = json.loads(raw[8:8 + size])
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__", {}), dict):
        raise ValueError("invalid safetensors metadata")
    return header, raw[8 + size:]


def preserve_artifact(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("delta must be a bounded regular file")
    meta = licence_metadata()
    if path.suffix == ".safetensors":
        raw = path.read_bytes()
        header, payload = tensor_parts(raw)
        header["__metadata__"] = {**header.get("__metadata__", {}), **meta}
        encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        stamped = struct.pack("<Q", len(encoded)) + encoded + payload
        before, after = tensor_parts(raw), tensor_parts(stamped)
        if before[1] != after[1] or {k: v for k, v in before[0].items() if k != "__metadata__"} != {k: v for k, v in after[0].items() if k != "__metadata__"}:
            raise ValueError("licence stamp changed tensor bytes or descriptors")
        path.write_bytes(stamped)
    elif path.suffix == ".pt":
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 256 or sum(i.file_size for i in infos) > MAX_BYTES:
                raise ValueError("Torch checkpoint ZIP ceiling")
            roots = [i.filename.removesuffix("data.pkl") for i in infos if i.filename.endswith("/data.pkl")]
            if len(roots) != 1 or len({i.filename for i in infos}) != len(infos):
                raise ValueError("ambiguous Torch checkpoint ZIP layout")
            member = roots[0] + "academic-licences.json"
            record = json.dumps(meta, ensure_ascii=True, separators=(",", ":")).encode()
            if member in archive.namelist():
                if archive.read(member) != record:
                    raise ValueError("checkpoint carries different terms")
                return
            original = {i.filename: (i.header_offset, i.CRC, i.file_size) for i in infos}
        temporary = path.with_name(path.name + ".licensed")
        try:
            shutil.copyfile(path, temporary)
            with zipfile.ZipFile(temporary, "a", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(member, record)
            with zipfile.ZipFile(temporary) as archive:
                if temporary.stat().st_size > MAX_BYTES or any((i.header_offset, i.CRC, i.file_size) != original[i.filename] for i in archive.infolist() if i.filename in original):
                    raise ValueError("licence stamp changed storage layout")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        raise ValueError("unsupported checkpoint container")
