# SPDX-License-Identifier: MIT
# Copyright (c) 2026 telegraphprotocol
def render(prompt: str, answer: str) -> str:
    """The node's RenderForJudgment (Telegraph pkg/wasm/tokenizer/render.go), character for
    character. ModernBERT has no chat template, so this is the only template."""
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"
