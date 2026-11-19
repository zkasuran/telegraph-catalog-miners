// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/ethereum/go-ethereum/crypto"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "one bounded delta path required")
		os.Exit(1)
	}
	info, err := os.Stat(os.Args[1])
	if err != nil || !info.Mode().IsRegular() || info.Size() > 64<<20 {
		fmt.Fprintln(os.Stderr, "bounded regular delta required")
		os.Exit(1)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	sum := sha256.Sum256(raw)
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{"sha256": hex.EncodeToString(sum[:]), "keccak": hex.EncodeToString(crypto.Keccak256(raw)), "bytes": len(raw)}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
