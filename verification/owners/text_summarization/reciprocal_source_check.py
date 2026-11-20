# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only original-task and matched batch-plan smoke."""
import json

from reciprocal import source_check

if __name__ == "__main__":
    print(json.dumps(source_check()))
