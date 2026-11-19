// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

import (
	"context"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/comparator"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

func TestPrimaryRawBodyAndTokenBoundary(t *testing.T) {
	for _, body := range []string{" {\"answer\":\"claim\",\"sources\":[\"https://example.test\"],\"brief\":\"claim\"} \r\n", "Brief only.\r\n"} {
		got := comparator.AdapterInput(context.Background(), "WEB_SEARCH_GENERAL", "source-only-proof", "a question", body)
		if got != body {
			t.Fatal("primary raw-body helper changed the literal string")
		}
	}
	x := make([]int64, 280)
	y := make([]int64, 280)
	for i := range x {
		x[i], y[i] = 7, 7
	}
	x[278], y[278] = 111, 222
	x[279], y[279] = 50282, 50282
	a, b := wasmrt.TruncateTokens(x, 256), wasmrt.TruncateTokens(y, 256)
	if len(a) != 256 || a[255] != 50282 {
		t.Fatal("primary closing SEP contract changed")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("lost late evidence must not be treated as distinguishable")
		}
	}
}
