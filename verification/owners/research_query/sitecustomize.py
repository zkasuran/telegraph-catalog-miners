# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only import barrier shared by the release gate and its subprocesses."""
import os
import sys


class SourceOnlyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"torch", "tensorflow", "onnxruntime", "transformers", "peft"}:
            raise RuntimeError("neural imports forbidden in the source-only release gate")
        return None


if os.environ.get("OWNER_SOURCE_ONLY") == "1":
    sys.meta_path.insert(0, SourceOnlyImports())
