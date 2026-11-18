# Evaluator registration integration test

This bounded workflow checks the exact evaluator packages in this repository
against Telegraph's live registration protocol. Each standard public GitHub
runner builds the pinned reference executor and loads one unchanged package on
the pinned foundation. A restricted SSH forward exposes it through an existing
TLS proxy for the node's registration checks.

The runner ends when the new registration receives a final decision or after
five hours. The endpoint exists for this integration test, not ongoing inference.
Private SSH keys are generated on the runner and never uploaded. Artifacts contain
the forwarding public key and final registration status only. The registration
wallet and provider credentials remain with the operator.

The matrix is manually dispatched and limited to the 13 recorded champion files.
The coordinator verifies each public endpoint before signing a registration.
The test does not guarantee a win on a new evaluation round.

## References and terms

- Reference kit, MIT, copyright 2026 telegraphprotocol:
  https://github.com/telegraphprotocol/telegraph-wasm-baseline/tree/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd
- The kit permits use, copying and modification subject to retaining its copyright
  and permission notice. Its original LICENSE remains in the downloaded test kit.
- Standard runners are free for public repositories:
  https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- GitHub Actions permits development and testing of the associated software:
  https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features

The repository's Source-Available No-Derivatives terms cover this verification
code. Package terms and bytes remain those of their original published copies.
