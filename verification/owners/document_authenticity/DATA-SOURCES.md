# Document evidence source boundary

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

## GovInfo [G1, G2]

Publisher: US Government Publishing Office.
Policy: https://www.govinfo.gov/about/policies
Grant: "By virtue of the foregoing, public documents can generally be reprinted without legal restriction."
Exclusion: "Publication in a Government document does not authorize any use or appropriation of such copyright material without consent of the owner."

The collector uses enrolled Congressional bill package identities and PDFs. The training inputs contain only package identity, byte hashes and our measured signature/range results. They contain no bill prose, images or third-party annotations. Downloaded PDFs are source evidence kept privately. They are not included in the publication patch. This is not a grant to all incorporated content.

Authentication guide: https://www.govinfo.gov/about/authentication
Primary distinctions: copied seals do not transfer signatures. Certificate expiration does not invalidate historical LTV by itself. Mathematical verification and whole-file coverage do not prove signing-time trust.

## Telegraph kit [K1, K2]

Revision: 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd.
K1: float_runtime/train_adapter.py, render, lines 236-239.
K2: pkg/scoring/comparator/response_format.go, AdapterInput, lines 347-377.
Licence: MIT, copyright 2026 telegraphprotocol. Grant includes "use, copy, modify, merge, publish, distribute, sublicense" subject to retaining the notice. No copied model or compiled adapter enters this build. The literal rendering contract is pinned in a test by reading the upstream pure function AST. The deployed caller is not verified by that test.

## Measurement and construction

Poppler pdfsig runs as an external installed program, not vendored code. Its stdout is an original local measurement. Certificate status remains uninterpreted. Each altered PDF is a named constructed tamper/padding control derived from an original. It is not a natural forgery annotation. Preferences are programmatically constructed by AI-assisted code. No human preference review is claimed.

The two research PDFs BILLS-117s66enr and changed-sample are excluded. There are no protected pool rows, model weights or development captures in new datasets.

Own source and constructed preferences use the checkout's reviewed SAND terms. Upstream grants remain unchanged. No neural loading, training or inference occurs locally.

## ModernBERT [BASE]

Answer.AI and LightOn, model card at revision 8949b909ec900327062f0ebf497f51aef5e6f0c8: https://huggingface.co/answerdotai/ModernBERT-base/raw/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md
Grant: "We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license."
Apache-2.0 section 2 grants a "copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form."
Full terms: https://www.apache.org/licenses/LICENSE-2.0.txt
Retain original terms, change notices and attribution. See verification/training/licences.json. The pretrained backbone is unchanged. No model files are downloaded by local commands.
Citation [BASE]: Warner et al. (2024), Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference, arXiv:2412.13663.
