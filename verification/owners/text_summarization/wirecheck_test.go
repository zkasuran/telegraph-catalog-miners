// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

import (
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func TestLiteralPythonWireRootMatchesPrimaryExpression(t *testing.T) {
	// Recorded by our literal Python wire implementation for this exact batch.
	const python = "c420ccf77f805ebe4c4a61e959de7a48e2650f93fcaa238592731c7b05f3fc36"
	actual := remote.BatchInputRoot([][]int64{{50281, 22, 50282}})
	if actual != python {
		t.Fatalf("literal primary wire root %q differs from Python %q", actual, python)
	}
}
