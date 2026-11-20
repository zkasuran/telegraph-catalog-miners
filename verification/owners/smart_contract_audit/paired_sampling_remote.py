# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual weighted sibling batch iterator and controls, authorized runner only."""
import json
from collections import Counter
from pathlib import Path

from paired_sampling import pairs
from policy import remote_only


class SiblingBatchSampler:
    def __init__(self, rows, frozen, seed):
        remote_only()
        self.rows = rows
        self.batches, self.weights = pairs(rows, frozen)
        self.seed = seed
        self.generator = None

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        remote_only()
        import torch
        from torch.utils.data import WeightedRandomSampler
        if self.generator is None:
            self.generator = torch.Generator().manual_seed(self.seed)
        for index in WeightedRandomSampler(self.weights, len(self.batches), replacement=True, generator=self.generator):
            yield self.batches[index]


def sampler(rows, frozen, hp):
    remote_only()
    if hp.batch_size != 2:
        raise ValueError("sibling batching requires the existing batch size two")
    return SiblingBatchSampler(rows, frozen, hp.seed)


def check(rows, frozen, output_root, seed):
    remote_only()
    from torch.utils.data import DataLoader
    a, b = SiblingBatchSampler(rows, frozen, seed), SiblingBatchSampler(rows, frozen, seed)
    first, second = list(a), list(b)
    if first != second or len(first) * 2 != len(rows):
        raise ValueError("actual sibling batch repeat/length control failed")
    loader = DataLoader(rows, batch_sampler=SiblingBatchSampler(rows, frozen, seed), collate_fn=lambda items: items)
    batches = list(loader)
    if len(batches) != len(first):
        raise ValueError("actual DataLoader batch contract changed")
    counts = Counter()
    for batch in batches:
        if len(batch) != 2 or {r["vulnerable"] for r in batch} != {True, False}:
            raise ValueError("actual minibatch did not contain both source labels")
        left, right = batch
        if left["sibling_group"] != right["sibling_group"] or left["response_view"] != right["response_view"] or left["chosen"] != right["rejected"] or left["rejected"] != right["chosen"]:
            raise ValueError("actual minibatch literal sibling contract failed")
        counts[left["template_group"]] += 1
    result = {"actual_batches": len(batches), "actual_rows": sum(map(len, batches)),
              "both_labels_each_actual_batch": True, "literal_preferences_reversed": True,
              "repeat_indices_identical": first == second, "actual_group_batch_counts": dict(counts),
              "optimizer_steps_per_epoch_at_accumulation_four": len(loader) // 4,
              "audit_used_for_fitting": False, "model_quality_claimed": False}
    (Path(output_root) / "paired-batch-runtime-controls.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
