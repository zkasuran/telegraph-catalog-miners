// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Use the upstream full parser with a fresh, verified live class binding.
// This changes only the isolated tool's in-process map, never a node/catalog DB.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
)

type classProof struct {
	Schema         int               `json:"schema_version"`
	Family         string            `json:"family"`
	Child          string            `json:"child"`
	Class          string            `json:"class"`
	FetchedAt      string            `json:"fetched_at"`
	URLs           map[string]string `json:"urls"`
	Resolve        json.RawMessage   `json:"family_resolve"`
	ChildClass     string            `json:"child_class"`
	Evaluation     json.RawMessage   `json:"child_evaluation"`
	CatalogChanged bool              `json:"controller_catalog_changed"`
}

func execute() error {
	pkg := flag.String("package", "", "exact checkpoint package")
	foundation := flag.String("foundation", "", "bound foundation layout")
	classes := flag.String("classes", "", "fresh verified live class proof")
	out := flag.String("out", "", "manifest output directory")
	flag.Parse()
	file, err := os.Open(*classes)
	if err != nil {
		return err
	}
	defer file.Close()
	var proof classProof
	decoder := json.NewDecoder(io.LimitReader(file, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&proof); err != nil {
		return err
	}
	if proof.Schema != 1 || proof.Family != "THREAT_IP_REPUTATION" || proof.Child != "THREAT_IP_REPUTATION_GENERAL" || proof.Class != "judgment" || proof.ChildClass != "judgment" || proof.CatalogChanged {
		return fmt.Errorf("live class identity mismatch, refusing to guess/override")
	}
	if proof.URLs["family"] != "https://devnode.telegraphprotocol.com/api/wasm?intent=THREAT_IP_REPUTATION&resolve=THREAT_IP_REPUTATION" || proof.URLs["child"] != "https://devnode.telegraphprotocol.com/engine/v1/intents/THREAT_IP_REPUTATION_GENERAL" {
		return fmt.Errorf("class authority URL mismatch")
	}
	fetched, err := time.Parse(time.RFC3339Nano, proof.FetchedAt)
	if err != nil {
		return err
	}
	age := time.Since(fetched)
	if age < -time.Minute || age > 15*time.Minute {
		return fmt.Errorf("live class proof stale/future")
	}
	var resolved map[string]interface{}
	if err := json.Unmarshal(proof.Resolve, &resolved); err != nil {
		return err
	}
	if resolved["intent"] != proof.Family || resolved["class"] != proof.Class || resolved["scorer"] != "judgment" || resolved["routed"] != true {
		return fmt.Errorf("live resolve does not support assigned judgment task")
	}
	var evaluation map[string]interface{}
	if err := json.Unmarshal(proof.Evaluation, &evaluation); err != nil {
		return err
	}
	if evaluation["kind"] != "judgment" {
		return fmt.Errorf("child evaluation disagrees")
	}
	// Supported upstream tool API mirrors the production DB-loaded path.
	catalog.UseClasses(map[string]catalog.Class{proof.Family: catalog.ClassJudgment, proof.Child: catalog.ClassJudgment})
	artifact, manifest, err := adapterbuild.CheckLoRA(context.Background(), adapterbuild.LoRAReq{
		Intent: proof.Family, PackagePath: *pkg, Class: proof.Class, FoundationPath: *foundation, OutDir: *out,
	}, func(format string, args ...interface{}) { fmt.Fprintf(os.Stderr, format+"\n", args...) })
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]interface{}{"sha256": artifact.SHA256, "keccak256": artifact.Keccak256, "bytes": artifact.Bytes, "manifest": manifest})
}

func main() {
	if err := execute(); err != nil {
		fmt.Fprintln(os.Stderr, "live-bound full parser: FAILED:", err)
		os.Exit(1)
	}
}
