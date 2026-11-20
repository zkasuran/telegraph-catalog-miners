# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Restore only the retained licensed best adapter on the public CPU runner."""
import struct
import zipfile

from contracts import file_sha, remote_only, sha, strict_json

PARENT_RUN = 44647112954
PARENT_COMMIT = "27ab80abc1b97ad1dba7b4f160ff90d796cce92e"
PARENT_CHECKPOINT_SHA = "2f3ce184bda5bf84c8a23d10b82d6bf34e378195f2d4da93320afa1b409000c4"
FAILED_TENSOR_SHA = "e0723e4c24864fc4d54d6bf28081702bad5f65e568926a19d079bbbae85c50b1"
FAILED_TENSOR_SHAS = {FAILED_TENSOR_SHA, "5d48fda32609747989ca64339906b91154eed0701d8ad9d76de002087286e313",
                      "c62751c03228a60574770785945276844fa2f1623b9a601673cd426b3b833caf",
                      "1542a65918e4b012456d135ec294978a33d44aee4894b317c704b980c4c70f26",
                      "4c3b393a69c2683db8ccf226d63539558666529e1d2fd56537056d9157362320"}


def check_candidate_tensor(raw):
    if len(raw) < 9 or len(raw) > 64 << 20:
        raise ValueError("candidate package byte ceiling")
    header_bytes = struct.unpack("<Q", raw[:8])[0]
    if not 0 < header_bytes < len(raw)-8 or header_bytes > 4 << 20:
        raise ValueError("candidate package header bound")
    digest = sha(raw[8+header_bytes:])
    if digest in FAILED_TENSOR_SHAS:
        raise ValueError("unchanged failed tensor payload must not be audited or registered again")
    return digest


def restore(root):
    remote_only()
    archive_path = root/"parent-checkpoint/contract-run/licensed-checkpoint.zip"
    if file_sha(archive_path) != PARENT_CHECKPOINT_SHA:
        raise ValueError("retained parent checkpoint digest mismatch")
    target = root/"retained-best-adapter"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        notice = strict_json(archive.read("CHECKPOINT-NOTICE.json"))
        records = {row["path"]: row for row in notice["files"]}
        for name in ("adapter_model.safetensors", "adapter_config.json", "reward_head.pt"):
            path = "trained-adapter/"+name
            if path not in records or archive.getinfo(path).file_size > 64 << 20:
                raise ValueError("retained adapter manifest/bound mismatch")
            raw = archive.read(path)
            import hashlib
            if hashlib.sha256(raw).hexdigest() != records[path]["sha256"]:
                raise ValueError("retained adapter tensor/config digest mismatch")
            (target/name).write_bytes(raw)
        for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "CC-BY-4.0.txt", "licences.json"):
            raw = archive.read(name)
            if not raw:
                raise ValueError("parent checkpoint did not retain grants")
            (target/("parent-"+name)).write_bytes(raw)
    return target
