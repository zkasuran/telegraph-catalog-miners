// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Source-only literal node-client test with constructed HTTP replies. No inference.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

type control struct {
	Name   string `json:"name"`
	Passed bool   `json:"passed"`
}

func run() error {
	var batches [][][]int64
	if err := json.NewDecoder(io.LimitReader(os.Stdin, 1<<20)).Decode(&batches); err != nil {
		return err
	}
	roots := make([]string, len(batches))
	for i, batch := range batches {
		roots[i] = remote.BatchInputRoot(batch)
	}
	tokens := [][]int64{{50281, 50282}, {1}, {2}, {3}, {4}, {5}}
	ident := remote.BatchIdentity{FoundationHash: strings.Repeat("a", 64), AdapterHash: strings.Repeat("b", 64), TokenizerHash: strings.Repeat("c", 64), InputRoot: remote.BatchInputRoot(tokens)}
	finite := []uint32{0, 2147483648, 1, 2139095039, 1065353216, 3212836864}
	checks := []control{}
	for _, name := range []string{"finite_exact_bits", "wrong_foundation_hash", "wrong_adapter_hash", "wrong_tokenizer_hash", "wrong_input_root", "nan_bits", "positive_inf_bits", "negative_inf_bits", "overflow_bits", "negative_bits", "decimal_score", "boolean_score", "wrong_result_count", "temporary_503", "module_422"} {
		wireError := ""
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body struct {
				Batch []struct {
					Tokens []int64 `json:"tokens"`
				} `json:"batch"`
				remote.BatchIdentity
			}
			if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" || json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body) != nil || len(body.Batch) != len(tokens) || body.BatchIdentity != ident {
				wireError = "literal primary caller payload differs"
				http.Error(w, wireError, 400)
				return
			}
			if name == "temporary_503" || name == "module_422" {
				status := 503
				if name == "module_422" {
					status = 422
				}
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"error":"constructed refusal"}`))
				return
			}
			results := []map[string]any{}
			for _, bit := range finite {
				results = append(results, map[string]any{"score": bit})
			}
			response := map[string]any{"results": results, "foundation_hash": ident.FoundationHash, "adapter_hash": ident.AdapterHash, "tokenizer_hash": ident.TokenizerHash, "input_root": ident.InputRoot}
			switch name {
			case "wrong_foundation_hash":
				response["foundation_hash"] = strings.Repeat("0", 64)
			case "wrong_adapter_hash":
				response["adapter_hash"] = strings.Repeat("0", 64)
			case "wrong_tokenizer_hash":
				response["tokenizer_hash"] = strings.Repeat("0", 64)
			case "wrong_input_root":
				response["input_root"] = strings.Repeat("0", 64)
			case "nan_bits":
				results[0]["score"] = uint64(2143289344)
			case "positive_inf_bits":
				results[0]["score"] = uint64(2139095040)
			case "negative_inf_bits":
				results[0]["score"] = uint64(4286578688)
			case "overflow_bits":
				results[0]["score"] = uint64(4294967296)
			case "negative_bits":
				results[0]["score"] = -1
			case "decimal_score":
				results[0]["score"] = 1.25
			case "boolean_score":
				results[0]["score"] = true
			case "wrong_result_count":
				response["results"] = results[:1]
			}
			if err := json.NewEncoder(w).Encode(response); err != nil {
				wireError = err.Error()
			}
		}))
		client := &remote.RemoteExecutor{URL: server.URL, Client: server.Client()}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		observed, echoed, err := client.ScoreBatchWithIdentity(ctx, tokens, &ident)
		cancel()
		server.Close()
		if wireError != "" {
			return errors.New(wireError)
		}
		var rejected *remote.RemoteRejectedError
		isRejected := errors.As(err, &rejected)
		if name == "finite_exact_bits" {
			if err != nil || !echoed || len(observed) != len(finite) {
				return fmt.Errorf("finite wire control failed: %v", err)
			}
			for i, value := range observed {
				if value.Score.Bits() != finite[i] {
					return errors.New("primary client rounded exact FP32 bits")
				}
			}
		} else {
			if err == nil {
				return fmt.Errorf("primary client accepted %s", name)
			}
			wantRejected := strings.HasPrefix(name, "wrong_") && name != "wrong_result_count" || name == "module_422"
			if isRejected != wantRejected {
				return fmt.Errorf("wrong module/transport verdict for %s: %v", name, err)
			}
		}
		checks = append(checks, control{Name: name, Passed: true})
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"input_roots": roots, "controls": checks, "neural_operations": false, "replies": "constructed HTTP fixtures, not engine scores"})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
