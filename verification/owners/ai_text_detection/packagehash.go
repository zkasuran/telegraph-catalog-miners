// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

import (
	"encoding/hex"
	"fmt"
	"github.com/ethereum/go-ethereum/crypto"
	"os"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "one package path required")
		os.Exit(1)
	}
	info, err := os.Stat(os.Args[1])
	if err != nil || !info.Mode().IsRegular() || info.Size() > 64<<20 {
		fmt.Fprintln(os.Stderr, "bounded regular package required")
		os.Exit(1)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(hex.EncodeToString(crypto.Keccak256(raw)))
}
