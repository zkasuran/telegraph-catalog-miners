# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure retained generated-byte/producer/source/plan binding, no model imports."""
import json
from common import sha
from window_contract import validate_augmented


def validate_restored(raw, producer_raw, lineage, plan_raw, sources, source_raw):
    producer = json.loads(producer_raw)
    plan = json.loads(plan_raw)
    if (sha(raw) != lineage["data_sha256"] or sha(producer_raw) != lineage["producer_report_sha256"] or
            lineage["source_manifest_sha256"] != sha(source_raw) or
            lineage["plan_sha256"] != sha(plan_raw) or producer["plan_sha256"] != sha(plan_raw) or
            producer["run_id"] != lineage["producer_run_id"] or producer["commit"] != lineage["producer_commit"] or
            producer.get("audit_used") is not False):
        raise ValueError("locked actual augmentation byte/producer/source/plan identity mismatch")
    return validate_augmented(raw, plan, sources, producer), producer
