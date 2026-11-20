# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Materialize the reviewed public harness bytes in this standalone namespace."""
from contracts import HERE, REPO, fetch

COMMIT = "063041289cd2c035c5fee8cc604319230f30812d"
FILES = {
    "common.py": "dbd33680401b5e98afe97dfb1e495da294ffdc124c0857c31168697968234784",
    "proxy.py": "636c6e8b6e87d6a368c2719d2281d39f2d20da3cb7a3cd51da8c10a697aa0a51",
    "stagecheck.go": "0d8b92f16ee6241208738888313ce1db11ead5d9bba4cfbfb53b37d2f0d67a3b",
    "stagecheck_test.go": "c530c0da60b902ad4627d13e9efd10117e8e54ca317493c16d6a0d17109c87ec",
}


def main():
    for name, expected in FILES.items():
        fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/verification/registration/{name}",
              HERE / name, expected, 1 << 20)
    fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/verification/training/requirements.lock",
          HERE / "requirements.lock", "4c2b80fd53899579cccb949ef49760b5f53d32cb00e2e3c85bdc857345c832ac", 1 << 20)
    print("PINNED_STANDALONE_HARNESS_MATERIALIZED 4")


if __name__ == "__main__":
    main()
