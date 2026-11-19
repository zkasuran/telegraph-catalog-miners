// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Compile and exercise the actual primary wire equality. No model is loaded.
package main

import (
	"fmt"
	"os"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func main() {
	a := remote.BatchIdentity{FoundationHash: "aabb", AdapterHash: "ccdd", TokenizerHash: "eeff"}
	b := remote.BatchIdentity{FoundationHash: "AABB", AdapterHash: "CCDD", TokenizerHash: "EEFF"}
	if !a.Equal(b) {
		os.Exit(1)
	}
	b.AdapterHash = "0xCCDD"
	if a.Equal(b) {
		os.Exit(2)
	}
	b.AdapterHash = "wrong"
	if a.Equal(b) {
		os.Exit(3)
	}
	fmt.Println("PRIMARY WIRE PASS: casefold accepted, prefixed hash and wrong hash refused; no model operations")
}
