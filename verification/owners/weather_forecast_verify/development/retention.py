# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Keep exact already-fitted bytes in the real uploader's bounded JSON root."""
import hashlib
import json
from pathlib import Path
import re

FILES = {"data/frozen-controls.json": "frozen-development-controls.json",
         "data/data-report.json": "development-source-report.json"}
FEATURES = {"length", "answer_only_numeric", "answer_only_digits", "overlap"}
MAX_BYTES = 2 << 20


def strict_json(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate retained artifact field")
            result[name] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite retained artifact value")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def validate_controls(value):
    if not isinstance(value, dict) or value.get("fit_split") != "train":
        raise ValueError("only an already train-fitted control snapshot may be retained")
    coefficients = value.get("integer_coefficients")
    if not isinstance(coefficients, dict) or set(coefficients) != FEATURES:
        raise ValueError("complete immutable shortcut coefficients required")
    if any(type(item) is not int or item not in (-1, 1) for item in coefficients.values()):
        raise ValueError("literal frozen integer control direction required")
    weights = value.get("answer_only_token_weights")
    if not isinstance(weights, dict) or len(weights) > 100000:
        raise ValueError("bounded answer-only control dictionary required")
    if any(not isinstance(name, str) or len(name.encode()) > 256 or type(weight) is not int or abs(weight) > (1 << 20)
           for name, weight in weights.items()):
        raise ValueError("invalid retained answer-only control")
    return True


def licensed_controls(controls, terms):
    validate_controls(controls)
    if not isinstance(terms, dict) or set(terms) != {"sand", "kit_mit", "apache_2_0"}:
        raise ValueError("retained complete component grants required")
    if any(not isinstance(value, str) or not value.strip() or len(value.encode()) > MAX_BYTES for value in terms.values()):
        raise ValueError("bounded nonempty licence text required")
    result = {**controls, "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "x-terms": dict(terms),
              "x-fact-terms": "NWS source facts are public domain and are not subject to our copyright; no endorsement implied"}
    json.dumps(result, allow_nan=False)
    return result


def immutable_write(path, raw):
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("preserve existing retained bytes; never replace a differing snapshot")
    else:
        with path.open("xb") as output:
            output.write(raw)


def retain_fitted_records(root, context, identity):
    root, context = Path(root), Path(context)
    if (not isinstance(identity, dict) or identity.get("family") != "WEATHER_FORECAST_VERIFY"
            or type(identity.get("run_id")) is not int or identity["run_id"] <= 0
            or not re.fullmatch(r"[a-f0-9]{40}", identity.get("commit", ""))):
        raise ValueError("actual run/commit identity required for retained artifacts")
    outputs, records = [], []
    for source_name, target_name in FILES.items():
        source = context / source_name
        if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
            raise ValueError("bounded regular fitted/source file required")
        raw = source.read_bytes()
        value = strict_json(raw)
        if source_name.endswith("frozen-controls.json"):
            validate_controls(value)
            if value.get("SPDX-License-Identifier") != "LicenseRef-zkasuran-SAND-1.0" or not value.get("x-terms"):
                raise ValueError("fitted control licence must be present in its bytes")
        elif not isinstance(value, dict) or value.get("independent_audit") is not False:
            raise ValueError("source report must remain development-only")
        outputs.append((root / target_name, raw))
        records.append({"path": target_name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    binding = {"schema_version": 1, "stage": "DEVELOPMENT_ARTIFACTS_RETAINED", "family": identity["family"],
               "run_id": identity["run_id"], "commit": identity["commit"], "files": records,
               "independent_audit": False, "registration_ready": False, "fitting_performed_by_retention": False}
    encoded = (json.dumps(binding, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # Validate any existing targets before writing even one new artifact.
    for path, raw in outputs + [(root / "development-artifact-binding.json", encoded)]:
        if path.exists() and (path.is_symlink() or not path.is_file() or path.read_bytes() != raw):
            raise ValueError("existing immutable retention output differs")
    for path, raw in outputs + [(root / "development-artifact-binding.json", encoded)]:
        immutable_write(path, raw)
    return binding


def verify_retained_records(root):
    root = Path(root)
    binding = strict_json((root / "development-artifact-binding.json").read_bytes())
    if (binding.get("schema_version") != 1 or binding.get("stage") != "DEVELOPMENT_ARTIFACTS_RETAINED"
            or binding.get("family") != "WEATHER_FORECAST_VERIFY"
            or binding.get("registration_ready") is not False or binding.get("independent_audit") is not False
            or binding.get("fitting_performed_by_retention") is not False):
        raise ValueError("retention binding cannot qualify an audit or registration")
    if (not isinstance(binding.get("files"), list) or len(binding["files"]) != len(FILES)
            or {record.get("path") for record in binding["files"]} != set(FILES.values())):
        raise ValueError("complete unique retention inventory required")
    for record in binding["files"]:
        name = record["path"]
        if name not in set(FILES.values()):
            raise ValueError("retention binding points outside the declared artifact root")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("retained artifact is absent or unsafe")
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("retained bytes changed after fitting")
    return binding
