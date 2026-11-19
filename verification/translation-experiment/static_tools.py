# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Generate the family-specialized source copy without executing any model."""
import ast
from pathlib import Path

from reuse_helpers import package_source

HERE = Path(__file__).resolve().parent


def main():
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    (HERE / "out/translation-packagecheck.go").write_text(package_source())
    print("Python syntax checked; unchanged controller parser copied with translation family constant")


if __name__ == "__main__":
    main()
