# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Own terms and upstream grants attached without changing tensor payloads."""
import hashlib
import json
import shutil
import struct
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWN_ID = "LicenseRef-zkasuran-SAND-1.0"
OWN_SHA = "e6cf0cc83fbc51616590209d71ef1fa5fe46c130bef44e0d97d439ee91663447"
RELEASE_SCOPE = (
    "Upstream scope: Wikidata structured facts retain CC0-1.0. Telegraph kit code\n"
    "retains MIT. ModernBERT retains Apache-2.0. Their grants are in NOTICE and the\n"
    "named sidecars. The unchanged SAND terms below cover our competitive additions\n"
    "including development-only training and source timeout recovery. Prior grants\n"
    "remain in force.\n\n"
)
APACHE_SHA = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
MIT_SHA = "e20635bc37e1ec73cc962c20cf23b0043bfcf20d01dbcefb959238ebc2f8fde1"
MAX_BYTES = 64 << 20
MAX_HEADER = 1 << 20
SIDECARS = ("LICENSE", "UPSTREAM-APACHE-2.0.txt", "KIT-LICENSE", "NOTICE", "SOURCE-CC0.txt")


def licence_metadata():
    own = (HERE / "LICENSE").read_text()
    apache = (HERE / "UPSTREAM-APACHE-2.0.txt").read_text()
    mit = (HERE / "KIT-LICENSE").read_text()
    cc0 = (HERE / "SOURCE-CC0.txt").read_text()
    if hashlib.sha256(cc0.encode()).hexdigest() != "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499":
        raise ValueError("full primary CC0 terms changed")
    if (not own.startswith(RELEASE_SCOPE) or hashlib.sha256(own[len(RELEASE_SCOPE):].encode()).hexdigest() != OWN_SHA
            or hashlib.sha256(apache.encode()).hexdigest() != APACHE_SHA):
        raise ValueError("own or upstream licence bytes differ from the reviewed terms")
    if hashlib.sha256(mit.encode()).hexdigest() != MIT_SHA:
        raise ValueError("upstream MIT terms differ from the reviewed bytes")
    return {"x-source-cc0": cc0, "x-upstream-apache-2.0": apache, "x-upstream-mit": mit,
            "x-upstream-notice": (HERE / "NOTICE").read_text(), "x-own-sand": own, "x-licence": OWN_ID}


def bounded_raw(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("checkpoint must be a bounded regular file")
    return path.read_bytes()


def tensor_parts(raw):
    if len(raw) < 8:
        raise ValueError("checkpoint safetensors header is truncated")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 0 < size <= MAX_HEADER or 8 + size > len(raw):
        raise ValueError("checkpoint safetensors header exceeds its ceiling")
    header = json.loads(raw[8:8 + size])
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__", {}), dict):
        # Container decoding has one malformed-input ValueError contract,
        # including JSON values of the wrong shape, not Python caller types.
        raise ValueError("invalid checkpoint metadata")  # noqa: TRY004
    return header, raw[8 + size:]


def stamp_safetensors(path):
    raw = bounded_raw(path)
    header, payload = tensor_parts(raw)
    metadata = {**header.get("__metadata__", {}), **licence_metadata()}
    if header.get("__metadata__") == metadata:
        return hashlib.sha256(payload).hexdigest()
    header["__metadata__"] = metadata
    encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    stamped = struct.pack("<Q", len(encoded)) + encoded + payload
    if len(encoded) > MAX_HEADER or len(stamped) > MAX_BYTES:
        raise ValueError("licensed checkpoint exceeds byte ceiling")
    temporary = path.with_name(path.name + ".licensed")
    try:
        temporary.write_bytes(stamped)
        before, after = tensor_parts(raw), tensor_parts(temporary.read_bytes())
        if before[1] != after[1] or {k: v for k, v in before[0].items() if k != "__metadata__"} != {
                k: v for k, v in after[0].items() if k != "__metadata__"}:
            raise ValueError("licence stamp changed tensor bytes or descriptors")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def stamp_torch_zip(path):
    """Append an inert record inside the existing PyTorch ZIP64 prefix.

    Original storage records, offsets and alignment are left unchanged.
    Primary format: docs.pytorch.org/docs/2.15/notes/serialization.html.
    Actual weights_only roundtrip is checked remotely before any audit scoring.
    """
    bounded_raw(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 256 or sum(info.file_size for info in infos) > MAX_BYTES:
            raise ValueError("checkpoint ZIP member ceiling")
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError("duplicate checkpoint ZIP member")
        roots = [name.removesuffix("data.pkl") for name in names if name.endswith("/data.pkl")]
        if len(roots) != 1:
            raise ValueError("checkpoint is not the pinned PyTorch ZIP layout")
        member = roots[0] + "qa-licences.json"
        record = json.dumps(licence_metadata(), ensure_ascii=True, separators=(",", ":")).encode()
        if member in names:
            if archive.read(member) != record:
                raise ValueError("checkpoint has different embedded licence terms")
            return
        original = {info.filename: (info.header_offset, info.CRC, info.file_size) for info in infos}
    temporary = path.with_name(path.name + ".licensed")
    try:
        shutil.copyfile(path, temporary)
        with zipfile.ZipFile(temporary, "a", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(member, record)
        if temporary.stat().st_size > MAX_BYTES:
            raise ValueError("licensed checkpoint ZIP exceeds byte ceiling")
        with zipfile.ZipFile(temporary) as archive:
            for info in archive.infolist():
                if info.filename in original and (info.header_offset, info.CRC, info.file_size) != original[info.filename]:
                    raise ValueError("licence stamp changed checkpoint storage layout")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint_metadata(path):
    wanted = licence_metadata()
    if path.suffix == ".safetensors":
        header, _ = tensor_parts(bounded_raw(path))
        actual = header.get("__metadata__", {})
    elif path.suffix == ".pt":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith("/qa-licences.json")]
            if len(members) != 1:
                raise ValueError("checkpoint lacks embedded licence record")
            actual = json.loads(archive.read(members[0]))
    else:
        raise ValueError("unsupported checkpoint container")
    if any(actual.get(k) != v for k, v in wanted.items()):
        raise ValueError("checkpoint does not carry complete own and upstream terms")


def preserve_artifact(path):
    if path.suffix == ".safetensors":
        stamp_safetensors(path)
    elif path.suffix == ".pt":
        stamp_torch_zip(path)
    verify_checkpoint_metadata(path)
