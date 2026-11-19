# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Read the real archive value for the CLI smoke, without guessing a number."""
from build import load_archive

if __name__ == "__main__":
    observations, _ = load_archive()
    print(observations[("USW00012839", "2025-09-01", "TMAX")].value)
