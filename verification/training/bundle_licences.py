# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Generate the embedded terms from the reviewed licence texts."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sand", required=True, type=Path)
    parser.add_argument("--mit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    request = urllib.request.Request("https://www.apache.org/licenses/LICENSE-2.0.txt",
                                     headers={"User-Agent": "Telegraph-training-check/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        apache = response.read(64 << 10).decode()
    if "Grant of Copyright License" not in apache or "END OF TERMS AND CONDITIONS" not in apache:
        raise ValueError("incomplete Apache licence response")
    bundle = {"notice": "ModernBERT by Answer.AI and LightOn, Apache-2.0. Architectures and pretrained weights: https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8. Telegraph reference kit, Copyright (c) 2026 telegraphprotocol, MIT, commit 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd. Our trained deltas and code, Copyright (c) 2026 zkasuran, LicenseRef-zkasuran-SAND-1.0. Third-party grants remain in force for their portions. Training sources: Kotzias, D. (2015), Sentiment Labelled Sentences, DOI 10.24432/C57604, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); Civil Comments and Jigsaw toxicity annotations, CC0-1.0 (https://www.tensorflow.org/datasets/catalog/civil_comments). Source labels retained; preference answers and label flips constructed by AI-assisted code. No licensor endorsement is implied.",
              "apache_2_0": apache, "kit_mit": args.mit.read_text(), "sand": args.sand.read_text()}
    args.out.write_text(json.dumps(bundle, indent=2) + "\n")
    print("Licence bundle generated: " + hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
