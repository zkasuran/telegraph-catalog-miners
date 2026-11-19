// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Source-only check against the counterparty's actual literal input-root code.
package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	remote "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func TestLiteralCounterpartyInputRoot(t *testing.T) {
	_, own, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("test source path unavailable")
	}
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(own), "input-root-vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Batch [][]int64 `json:"batch"`
		Root  string    `json:"root"`
	}
	if err := json.Unmarshal(raw, &vectors); err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 3 {
		t.Fatal("three independent fixed wire fixtures required")
	}
	for _, vector := range vectors {
		if got := remote.BatchInputRoot(vector.Batch); got != vector.Root {
			t.Fatalf("literal counterparty root differs: %s != %s", got, vector.Root)
		}
	}
}
