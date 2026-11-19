# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Reuse the verified remote trainer with broader frozen source contracts."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "development"))
from runtime_contract import remote_only
import remote_develop
from broad_data import FORMATS, MAX_TOKENS, build, control_accuracy, digest, render, source_controls


def controls(splits, tokenizer, root):
    observed = source_controls(splits["validation"])
    train = splits["train"]
    def length(prompt, answer):
        return len(tokenizer.encode(render(prompt, answer)).ids)
    fitted = control_accuracy(train, length)
    orientation = 1 if fitted >= .5 else -1
    frozen = {"schema_version": 1, "fitted_on": "train_only", "orientation": orientation,
              "train_sha256": digest((root / "data/train.jsonl").read_bytes()), "train_accuracy": max(fitted, 1-fitted),
              "baseline": "fixed token-count difference with immutable train-selected direction"}
    path = root / "frozen-token-length-baseline.json"
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    for style in FORMATS:
        rows = [r for r in splits["validation"] if r["format"] == style]
        observed[style]["train_fitted_token_length"] = control_accuracy(rows, lambda p, a: orientation * length(p, a))
        observed[style]["prompt_only"] = .5
    (root / "source-shortcut-report.json").write_text(json.dumps({"formats": observed, "token_baseline_sha256": digest(path.read_bytes()),
        "scope": "Measured development controls, not an independent audit or exhaustive semantic factual proof"}, indent=2) + "\n")
    return observed


def main():
    remote_only()
    config = json.loads((HERE / "config.json").read_text())
    if config["max_tokens"] != MAX_TOKENS or config["epochs"] != 3 or config["registration_ready"] is not False:
        raise ValueError("frozen broad development configuration mismatch")
    remote_develop.main(data_builder=build, max_tokens=MAX_TOKENS, control_hook=controls, development_mode="develop_broad")


if __name__ == "__main__":
    main()
