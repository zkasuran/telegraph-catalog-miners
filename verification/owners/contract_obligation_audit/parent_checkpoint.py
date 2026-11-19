# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Restore only the retained licensed best adapter on the public CPU runner."""
import zipfile

from contracts import file_sha, remote_only, strict_json

PARENT_RUN = 44564516958
PARENT_COMMIT = "e799c202149d6a7caec3c21ec1870d8be4f5b996"
PARENT_CHECKPOINT_SHA = "b703334e93dcc0ce5c163d14b5d5cf57aa42218b5ef46bbcf4570433f90380c0"
FAILED_TENSOR_SHA = "e0723e4c24864fc4d54d6bf28081702bad5f65e568926a19d079bbbae85c50b1"


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
