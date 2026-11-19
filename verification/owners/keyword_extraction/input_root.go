// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Input roots from the literal primary runtime. This never loads a model.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func main() {
	var batch [][]int64
	raw, err := io.ReadAll(io.LimitReader(os.Stdin, (1<<20)+1))
	if err != nil || len(raw) > 1<<20 {
		fmt.Fprintln(os.Stderr, "input byte ceiling")
		os.Exit(1)
	}
	if err := json.Unmarshal(raw, &batch); err != nil || len(batch) > 100 {
		fmt.Fprintln(os.Stderr, "invalid token batch")
		os.Exit(1)
	}
	for _, ids := range batch {
		if len(ids) > 512 {
			fmt.Fprintln(os.Stderr, "token ceiling")
			os.Exit(1)
		}
		for _, id := range ids {
			if id < 0 {
				fmt.Fprintln(os.Stderr, "negative token")
				os.Exit(1)
			}
		}
	}
	fmt.Println(remote.BatchInputRoot(batch))
}
