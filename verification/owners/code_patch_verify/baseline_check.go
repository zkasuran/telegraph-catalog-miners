// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Full parser execution for an untrained source control, not a candidate gate.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/ethereum/go-ethereum/crypto"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

func run() error {
	path := flag.String("package", "", "untrained control package")
	expected := flag.String("sha256", "", "actual external package digest")
	foundation := flag.String("foundation", "", "released pinned layout")
	out := flag.String("out", "", "private parser result")
	flag.Parse()
	info, err := os.Stat(*path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 64<<20 {
		return fmt.Errorf("bounded regular control package required")
	}
	raw, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	hash := sha256.Sum256(raw)
	if hex.EncodeToString(hash[:]) != *expected {
		return fmt.Errorf("control package hash mismatch")
	}
	layout, err := os.ReadFile(*foundation)
	if err != nil || len(layout) > 65536 {
		return fmt.Errorf("bounded released foundation layout required")
	}
	var spec wasmrt.FoundationSpec
	if err := json.Unmarshal(layout, &spec); err != nil {
		return err
	}
	adapter, err := wasmrt.ParseLoRAAdapter(raw, wasmrt.LoRAExpect{
		FoundationHash: "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
		TokenizerHash: "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
		Class: "judgment", Spec: spec, MaxBytes: 64 << 20,
	})
	if err != nil {
		return err
	}
	if adapter.Header.Intent != "CODE_PATCH_VERIFY" || adapter.Header.Training != nil ||
		adapter.Header.Metadata["x-cpv-role"] != "UNTRAINED_EXECUTION_CONTROL_NEVER_REGISTER" {
		return fmt.Errorf("not the explicit untrained control role")
	}
	result := map[string]any{"package_sha256": *expected, "package_keccak": hex.EncodeToString(crypto.Keccak256(raw)),
		"package_bytes": len(raw), "full_parser_passed": true, "training_record_present": false,
		"registration_ready": false, "role": "UNTRAINED_EXECUTION_CONTROL_NEVER_REGISTER"}
	encoded, err := json.Marshal(result)
	if err != nil {
		return err
	}
	return os.WriteFile(*out, append(encoded, '\n'), 0600)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "untrained control parser:", err)
		os.Exit(1)
	}
}
