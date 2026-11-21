# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded remote development executor. Importing this module never loads a model."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
FAMILY_ROOT = HERE.parent
sys.path.insert(0, str(FAMILY_ROOT))
import reference  # noqa: E402
from core import identity  # noqa: E402
from build import control_accuracy, sha, surface_accuracy, surface_tokens, write_json  # noqa: E402
from support import load_inputs, training_representations, view_row  # noqa: E402
from retention import licensed_controls, retain_fitted_records, verify_retained_records  # noqa: E402


def remote_only():
    reference.remote_only()
    if os.environ.get("WEATHER_DEVELOPMENT_ONLY") != "true":
        raise RuntimeError("separate development mode must be explicitly selected")


def frozen_controls(rows):
    remote_only()
    positive = control_accuracy(rows["train"], {name: 1 for name in ("length", "answer_only_numeric", "answer_only_digits", "overlap")})
    coefficients = {name: 1 if value["wins_half_units"] >= value["pairs"] else -1 for name, value in positive.items()}
    weights = Counter()
    for row in rows["train"]:
        weights.update(surface_tokens(row["chosen"]))
        weights.subtract(surface_tokens(row["rejected"]))
    weights = {name: weight for name, weight in sorted(weights.items()) if weight}
    validation = control_accuracy(rows["validation"], coefficients)
    validation["answer_only_bag_of_words"] = surface_accuracy(rows["validation"], weights)
    by_view = {}
    for view in ("json", "celsius", "bare"):
        projected = [view_row(row, view) for row in rows["validation"]]
        by_view[view] = control_accuracy(projected, coefficients)
        by_view[view]["answer_only_bag_of_words"] = surface_accuracy(projected, weights)
    fitted = {"fit_split": "train", "integer_coefficients": coefficients, "answer_only_token_weights": weights,
              "validation": validation, "validation_by_view": by_view,
              "policy": "fit once on actual training representations; retain immutable output; never refit to reproduce a selected package"}
    terms = {name: (FAMILY_ROOT / filename).read_text() for name, filename in
             (("sand", "LICENSE"), ("kit_mit", "KIT-LICENSE"), ("apache_2_0", "UPSTREAM-APACHE-2.0.txt"))}
    return licensed_controls(fitted, terms)


def run(root):
    remote_only()
    spec, rows, report = load_inputs()
    root.mkdir(parents=True, exist_ok=False)
    context = root / "development-context"
    shutil.copytree(HERE, context, ignore=shutil.ignore_patterns("*.py", "__pycache__", "verify.sh", "README.md"))
    for name in ("LICENSE", "KIT-LICENSE", "UPSTREAM-APACHE-2.0.txt"):
        shutil.copyfile(FAMILY_ROOT / name, context / name)
    shutil.copyfile(context / "data/train.jsonl", context / "data/source-train.jsonl")
    rows["train"] = training_representations(rows["train"])
    actual_train = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows["train"]).encode()
    (context / "data/train.jsonl").write_bytes(actual_train)
    report = {**report, "training_representation_sha256": sha(actual_train),
              "training_answer_views": dict(Counter(row["answer_view"] for row in rows["train"])),
              "representations_are_new_independent_examples": False}
    write_json(context / "data/data-report.json", report)
    controls = frozen_controls(rows)
    write_json(context / "data/frozen-controls.json", controls)
    retain_fitted_records(root, context, {"family": "WEATHER_FORECAST_VERIFY", "run_id": int(os.environ["GITHUB_RUN_ID"]),
                                          "commit": os.environ["GITHUB_SHA"]})
    if any(value["accuracy"] > 0.65 for view in controls["validation_by_view"].values() for value in view.values()):
        raise ValueError("train-fitted development shortcut exceeds the frozen source gate in an answer view")
    from live_class import capture_classes
    classes = capture_classes(root)
    current = json.loads(classes["child"]["raw"])
    expected = json.loads((FAMILY_ROOT / "sources/intent-projection.json").read_text())
    if current.get("request") != expected["request"] or current.get("response") != expected["response"]:
        raise ValueError("literal live development task changed")
    write_json(context / "live-preflight.json", {"class": "judgment", "schema_matches_frozen_task": True,
                                                "public_definition_only": True, "source_snapshot_sha256": sha((root / "class-snapshot.json").read_bytes())})
    driver_spec = importlib.util.spec_from_file_location("weather_development_driver", HERE / "driver.py")
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)
    driver.HERE = context
    driver.CONTRACT = json.loads((context / "contract.json").read_text())
    driver.FAMILY = driver.CONTRACT["family"]
    frozen = {str(path.relative_to(context)): sha(path.read_bytes()) for path in context.rglob("*") if path.is_file()}
    write_json(root / "development-context-binding.json", frozen)
    def bound_splits():
        for name, expected_sha in frozen.items():
            if sha((context / name).read_bytes()) != expected_sha:
                raise ValueError("train-fitted development context changed")
        for split, values in rows.items():
            if len({identity(row["prompt"]) for row in values}) != len(values):
                raise ValueError("development question overlap after context freeze")
        return report, rows, controls
    driver.read_bound_splits = bound_splits
    kit = reference.source_kit(root)
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    evaluate = trainer.evaluate
    history = []
    negatives = {view: {view_row(row, view)["rejected"].strip() for row in load_inputs()[1]["train"]}
                 for view in ("json", "celsius", "bare")}
    def format_evaluate(model, tokenizer, pairs, hyperparameters, *args, **kwargs):
        observed = {}
        for view in ("json", "celsius", "bare"):
            positional = list(args)
            named = dict(kwargs)
            if positional:
                positional[0] = negatives[view]
            else:
                named["train_negatives"] = negatives[view]
            observed[view] = evaluate(model, tokenizer, [view_row(row, view) for row in pairs], hyperparameters, *positional, **named)
        history.append(observed)
        write_json(root / "native-validation-history.json", {"history": history, "scope": "native Torch development validation, not reference-engine audit"})
        return observed["json"]
    trainer.evaluate = format_evaluate
    try:
        driver.train(root)
    except Exception:
        failure = root / "result.json"
        if failure.exists():
            actual_failure = json.loads(failure.read_text())
            actual_failure.update(status="DEVELOPMENT_ONLY_FAILURE", independent_audit=False, registration_ready=False)
            write_json(failure, actual_failure)
        raise
    finally:
        trainer.evaluate = evaluate
    actual = json.loads((root / "result.json").read_text())
    if not history or set(history[-1]) != {"json", "celsius", "bare"}:
        raise ValueError("actual native development validation views are missing")
    actual.update(status="DEVELOPMENT_ONLY", source_input_contract_sha256=sha((HERE / "input-contract.json").read_bytes()),
                  independent_audit=False, reference_engine_passed=False, registration_ready=False,
                  native_validation_views=history[-1],
                  train_fitted_controls_sha256=sha((root / "frozen-development-controls.json").read_bytes()),
                  development_artifact_binding_sha256=sha((root / "development-artifact-binding.json").read_bytes()))
    verify_retained_records(root)
    write_json(root / "result.json", actual)
    if (root / "registration-ready.json").exists():
        raise ValueError("development cannot emit registration readiness")


def main():
    remote_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
