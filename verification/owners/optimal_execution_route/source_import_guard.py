# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Enforce the source-only import boundary without loading any model."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


class SourceOnly:
    blocked = frozenset({"torch", "tensorflow", "onnxruntime", "transformers", "peft"})

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.blocked:
            raise RuntimeError("neural imports forbidden in the source-only release gate")
        return None


def prepare_guard():
    directory = HERE / ".gate-work/source-guard"
    if directory.is_symlink() or not directory.resolve().is_relative_to(HERE):
        raise ValueError("linked or escaped source guard directory")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    if path.is_symlink():
        raise ValueError("linked source guard file")
    path.write_text("# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\n"
                    "import sys\nfrom source_import_guard import SourceOnly\n"
                    "sys.meta_path.insert(0, SourceOnly())\n")


if __name__ == "__main__":
    prepare_guard()
