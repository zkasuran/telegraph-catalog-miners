// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "one package path required")
		os.Exit(2)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	var batch [][]int64
	if err = json.NewDecoder(io.LimitReader(os.Stdin, 1<<20)).Decode(&batch); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if len(batch) > 100 {
		os.Exit(2)
	}
	if err = json.NewEncoder(os.Stdout).Encode(map[string]string{
		"input_root": remote.BatchInputRoot(batch), "adapter_hash": hex.EncodeToString(crypto.Keccak256(raw)),
	}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}
