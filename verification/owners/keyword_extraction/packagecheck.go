// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Verify this family's live class and retain the primary static LoRA checker.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
)

func run() error {
	snapshot := flag.String("snapshot", "", "fresh bounded public metadata snapshot")
	packagePath := flag.String("package", "", "package for static checking only")
	foundation := flag.String("foundation", "", "pinned foundation layout")
	out := flag.String("out", "", "output directory")
	classOnly := flag.Bool("class-only", false, "source class preflight only, not a package verification")
	flag.Parse()
	raw, err := os.ReadFile(*snapshot)
	if err != nil {
		return err
	}
	if len(raw) > 2<<20 {
		return fmt.Errorf("metadata snapshot exceeds ceiling")
	}
	var s struct {
		At        time.Time `json:"at"`
		URL       string    `json:"url"`
		FamilyURL string    `json:"family_url"`
		Parent    struct {
			Resolve struct {
				Intent string `json:"intent"`
				Class  string `json:"class"`
			} `json:"resolve"`
		} `json:"parent"`
		Child struct {
			Intent     string `json:"intent_id"`
			Name       string `json:"intent_name"`
			Class      string `json:"class"`
			Evaluation struct {
				Kind string `json:"kind"`
			} `json:"evaluation"`
		} `json:"child"`
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		return err
	}
	if s.URL != "https://devnode.telegraphprotocol.com/engine/v1/intents/KEYWORD_EXTRACTION_GENERAL" ||
		s.FamilyURL != "https://devnode.telegraphprotocol.com/api/wasm?resolve=KEYWORD_EXTRACTION" ||
		s.Parent.Resolve.Intent != "KEYWORD_EXTRACTION" || s.Parent.Resolve.Class != "judgment" ||
		s.Child.Intent != "KEYWORD_EXTRACTION_GENERAL" || s.Child.Name != "KEYWORD_EXTRACTION_GENERAL" || s.Child.Class != "judgment" || s.Child.Evaluation.Kind != "judgment" ||
		time.Since(s.At) > 15*time.Minute || time.Until(s.At) > time.Minute {
		return fmt.Errorf("live child identity, class or freshness mismatch")
	}
	// This is scoped to the verified family in this isolated checker process.
	// It does not modify the controller catalog or skip tensor/finiteness gates.
	catalog.UseClasses(map[string]catalog.Class{"KEYWORD_EXTRACTION": catalog.ClassJudgment, "KEYWORD_EXTRACTION_GENERAL": catalog.ClassJudgment})
	defer catalog.ClearClasses()
	c, err := adapterbuild.ResolveClass("KEYWORD_EXTRACTION", "judgment")
	if err != nil {
		return err
	}
	if *classOnly {
		h := sha256.Sum256(raw)
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"family": "KEYWORD_EXTRACTION", "class": c,
			"snapshot_sha256": hex.EncodeToString(h[:]), "stage1_checked": false, "model_loaded": false})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	art, manifest, err := adapterbuild.CheckLoRA(ctx, adapterbuild.LoRAReq{
		Intent: "KEYWORD_EXTRACTION", PackagePath: *packagePath, Class: c,
		FoundationPath: *foundation, OutDir: *out,
	}, nil)
	if err != nil {
		return err
	}
	if !manifest.Verification.Stage1 || manifest.Verification.Loaded {
		return fmt.Errorf("static check scope mismatch")
	}
	pkg, err := os.ReadFile(*packagePath)
	if err != nil {
		return err
	}
	h := sha256.Sum256(pkg)
	if art.SHA256 != hex.EncodeToString(h[:]) {
		return fmt.Errorf("checked package SHA mismatch")
	}
	// Derive the literal runnable hash from bytes like LoRARuntime.Identity.
	// adapterbuild's registration hash is prefixed. Do not compare them by
	// stripping prefixes in the wire checker.
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"manifest": manifest, "sha256": art.SHA256,
		"registration_keccak256": art.Keccak256, "runtime_adapter_hash": hex.EncodeToString(crypto.Keccak256(pkg))})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
