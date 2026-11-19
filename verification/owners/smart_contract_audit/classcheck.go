// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Read-only full parser with live HTTPS class authority, not a class override.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
)

const intent = "SMART_CONTRACT_AUDIT"
const child = "SMART_CONTRACT_AUDIT_GENERAL"

type source struct { URL string `json:"url"`; Raw string `json:"raw"` }
type snapshot struct { Schema int `json:"schema"`; Fetched string `json:"fetched_at"`; Family source `json:"family"`; Child source `json:"child"` }

func boundRead(path, expected string, ceiling int64) ([]byte, error) {
	if !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(expected) { return nil, errors.New("full lowercase external hash required") }
	f, err := os.Open(path); if err != nil { return nil, err }; defer f.Close()
	info, err := f.Stat(); if err != nil || !info.Mode().IsRegular() || info.Size() > ceiling { return nil, errors.New("bounded regular file required") }
	raw, err := io.ReadAll(io.LimitReader(f, ceiling+1)); if err != nil || int64(len(raw)) > ceiling { return nil, errors.New("file ceiling") }
	hash := sha256.Sum256(raw); if hex.EncodeToString(hash[:]) != expected { return nil, errors.New("source bytes differ from external hash") }; return raw, nil
}

func verifySource(raw []byte) (catalog.Class, error) {
	var s snapshot
	d := json.NewDecoder(bytes.NewReader(raw)); d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil { return "", err }
	var extra any
	if d.Decode(&extra) != io.EOF || s.Schema != 1 || s.Family.URL != "https://devnode.telegraphprotocol.com/api/wasm?resolve="+intent || s.Child.URL != "https://devnode.telegraphprotocol.com/engine/v1/intents/"+child { return "", errors.New("exact live class source envelope required") }
	fetched, err := time.Parse(time.RFC3339Nano, s.Fetched)
	if err != nil || time.Since(fetched) > 15*time.Minute || time.Until(fetched) > time.Minute { return "", errors.New("stale or future class evidence") }
	var p struct { Resolve struct { Intent string `json:"intent"`; Class catalog.Class `json:"class"` } `json:"resolve"` }
	var c struct { ID string `json:"intent_id"`; Name string `json:"intent_name"`; Class catalog.Class `json:"class"`; Evaluation struct { Kind string `json:"kind"` } `json:"evaluation"` }
	if json.Unmarshal([]byte(s.Family.Raw), &p) != nil || json.Unmarshal([]byte(s.Child.Raw), &c) != nil { return "", errors.New("class payload JSON refused") }
	if p.Resolve.Intent != intent || p.Resolve.Class != catalog.ClassJudgment || c.ID != child || c.Name != child || c.Class != p.Resolve.Class || c.Evaluation.Kind != string(c.Class) { return "", errors.New("actual family, selected child and model class must agree") }
	// Miner-schema readiness is separate from scoring class. No inference about
	// the hidden rubric or permission to register follows from this class check.
	return c.Class, nil
}

func run() error {
	snapshotPath := flag.String("class-snapshot", "", "actual live source envelope")
	snapshotHash := flag.String("snapshot-sha256", "", "external envelope hash")
	packagePath := flag.String("package", "", "unchanged package")
	packageHash := flag.String("sha256", "", "external frozen package hash")
	foundationPath := flag.String("foundation", "", "pinned layout")
	foundationHash := flag.String("foundation-sha256", "", "external layout hash")
	out := flag.String("out", "", "bounded result directory")
	flag.Parse()
	if flag.NArg() != 0 || *out == "" { return errors.New("named arguments and output required") }
	raw, err := boundRead(*snapshotPath, *snapshotHash, 2<<20); if err != nil { return err }
	cls, err := verifySource(raw); if err != nil { return err }
	if _, err = boundRead(*packagePath, *packageHash, 64<<20); err != nil { return err }
	if _, err = boundRead(*foundationPath, *foundationHash, 1<<20); err != nil { return err }
	// This uses the kit's supported loaded-class path in one private checker
	// process. No compiled catalog, held adapter or live service is changed.
	catalog.UseClasses(map[string]catalog.Class{intent: cls, child: cls}); defer catalog.ClearClasses()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second); defer cancel()
	art, manifest, err := adapterbuild.CheckLoRA(ctx, adapterbuild.LoRAReq{Intent: intent, PackagePath: *packagePath, Class: string(cls), FoundationPath: *foundationPath, OutDir: *out}, nil)
	if err != nil { return err }
	if art.SHA256 != *packageHash || !manifest.Verification.Stage1 || manifest.Verification.Loaded { return errors.New("full static parser identity mismatch") }
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"sha256": art.SHA256, "keccak256": art.Keccak256, "bytes": art.Bytes, "manifest": manifest, "class_snapshot_sha256": *snapshotHash, "full_parser_passed": true, "model_scored": false})
}

func main() { if err := run(); err != nil { fmt.Fprintln(os.Stderr, "classcheck: FAILED:", err); os.Exit(1) } }
