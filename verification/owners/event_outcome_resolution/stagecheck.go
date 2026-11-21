// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Static validation using the unmodified pinned Telegraph CheckLoRA.
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
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

type source struct {
	URL string `json:"url"`
	Raw string `json:"raw"`
}
type snapshot struct {
	Schema    int    `json:"schema"`
	FetchedAt string `json:"fetched_at"`
	Family    source `json:"family"`
	Child     source `json:"child"`
}

func boundRead(path, expected string, limit int64) ([]byte, error) {
	if !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(expected) {
		return nil, errors.New("external lowercase SHA-256 required")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > limit {
		return nil, errors.New("bounded regular file required")
	}
	raw, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(raw)) > limit {
		return nil, errors.New("input byte ceiling exceeded")
	}
	h := sha256.Sum256(raw)
	if hex.EncodeToString(h[:]) != expected {
		return nil, errors.New("input SHA-256 mismatch")
	}
	return raw, nil
}

func checkSource(raw []byte, intent, child, expected string) (catalog.Class, error) {
	return checkScopedSource(raw, intent, child, expected, "general")
}

func checkScopedSource(raw []byte, intent, child, expected, scope string) (catalog.Class, error) {
	if !regexp.MustCompile(`^[A-Z][A-Z0-9_]{1,79}$`).MatchString(intent) ||
		(scope != "general" && scope != "parent_only") ||
		(scope == "general" && child != intent+"_GENERAL") || (scope == "parent_only" && child != intent) {
		return "", errors.New("exact family and selected child required")
	}
	var envelope snapshot
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&envelope); err != nil {
		return "", err
	}
	var extra any
	if d.Decode(&extra) != io.EOF || envelope.Schema != 1 ||
		envelope.Family.URL != "https://devnode.telegraphprotocol.com/api/wasm?resolve="+intent ||
		envelope.Child.URL != "https://devnode.telegraphprotocol.com/engine/v1/intents/"+child {
		return "", errors.New("source snapshot schema or URL mismatch")
	}
	t, err := time.Parse(time.RFC3339Nano, envelope.FetchedAt)
	if err != nil || time.Since(t) > 15*time.Minute || time.Until(t) > time.Minute {
		return "", errors.New("stale or future class snapshot")
	}
	var parent struct {
		Resolve struct {
			Intent string        `json:"intent"`
			Class  catalog.Class `json:"class"`
		} `json:"resolve"`
	}
	var descendant struct {
		ID     string        `json:"intent_id"`
		Name   string        `json:"intent_name"`
		Class  catalog.Class `json:"class"`
		Prefix bool          `json:"prefix"`
		Eval   struct {
			Kind string `json:"kind"`
		} `json:"evaluation"`
		Request struct {
			Family string `json:"served_by"`
		} `json:"request"`
		Response struct {
			Family string `json:"served_by"`
		} `json:"response"`
		Onchain struct {
			Family  string `json:"served_by"`
			Ready   bool   `json:"ready"`
			Version int    `json:"format_version"`
			Request []struct {
				Name     string `json:"name"`
				Type     string `json:"type"`
				Slot     string `json:"slot"`
				Index    int    `json:"index"`
				MaxBytes int    `json:"max_bytes"`
			} `json:"request"`
			Response []struct {
				Name     string   `json:"name"`
				Type     string   `json:"type"`
				Slot     string   `json:"slot"`
				Index    int      `json:"index"`
				MaxBytes int      `json:"max_bytes"`
				Allowed  []string `json:"allowed"`
			} `json:"response"`
		} `json:"onchain"`
	}
	if json.Unmarshal([]byte(envelope.Family.Raw), &parent) != nil || json.Unmarshal([]byte(envelope.Child.Raw), &descendant) != nil {
		return "", errors.New("invalid class source JSON")
	}
	c := parent.Resolve.Class
	if (c != catalog.ClassJudgment && c != catalog.ClassHybrid) || string(c) != expected ||
		parent.Resolve.Intent != intent || descendant.ID != child || descendant.Name != child ||
		descendant.Class != c || descendant.Request.Family != intent ||
		descendant.Response.Family != intent || descendant.Onchain.Family != intent || !descendant.Onchain.Ready {
		return "", errors.New("live source family, child or class mismatch")
	}
	if scope == "general" && descendant.Eval.Kind != string(c) {
		return "", errors.New("concrete child evaluation class mismatch")
	}
	if scope == "parent_only" && (!descendant.Prefix || (descendant.Eval.Kind != "" && descendant.Eval.Kind != string(c))) {
		return "", errors.New("explicit parent-only prefix source required")
	}
	if scope == "parent_only" {
		wire := descendant.Onchain
		if wire.Version != 4 || len(wire.Request) != 1 || len(wire.Response) != 2 {
			return "", errors.New("parent-only literal v4 wire required")
		}
		request, outcome, resolved := wire.Request[0], wire.Response[0], wire.Response[1]
		if request.Name != "event" || request.Type != "string" || request.Slot != "strings" || request.Index != 0 || request.MaxBytes != 256 ||
			outcome.Name != "outcome" || outcome.Type != "enum" || outcome.Slot != "strings" || outcome.Index != 0 || outcome.MaxBytes != 256 ||
			len(outcome.Allowed) != 3 || outcome.Allowed[0] != "yes" || outcome.Allowed[1] != "no" || outcome.Allowed[2] != "unresolved" ||
			resolved.Name != "resolved_yes" || resolved.Type != "bool" || resolved.Slot != "bools" || resolved.Index != 0 {
			return "", errors.New("parent-only field/slot contract mismatch")
		}
	}
	return c, nil
}

func run() error {
	intent := flag.String("intent", "", "exact candidate family")
	child := flag.String("child", "", "selected source child")
	expected := flag.String("expected-class", "", "class assertion, never an override")
	scope := flag.String("scope", "general", "explicit general or parent_only training-record scope")
	snapshotPath := flag.String("class-snapshot", "", "trusted HTTPS source envelope")
	snapshotSHA := flag.String("snapshot-sha256", "", "external source envelope hash")
	pkgPath := flag.String("package", "", "unchanged package")
	pkgSHA := flag.String("sha256", "", "external verified package hash")
	fndPath := flag.String("foundation", "", "pinned layout")
	fndSHA := flag.String("foundation-sha256", "", "external layout hash")
	out := flag.String("out", "", "private output directory")
	flag.Parse()
	if flag.NArg() != 0 || *out == "" {
		return errors.New("output directory required, no positional arguments")
	}
	raw, err := boundRead(*snapshotPath, *snapshotSHA, 2<<20)
	if err != nil {
		return err
	}
	c, err := checkScopedSource(raw, *intent, *child, *expected, *scope)
	if err != nil {
		return err
	}
	catalog.UseClasses(map[string]catalog.Class{*intent: c, *child: c})
	defer catalog.ClearClasses()
	if _, err := boundRead(*fndPath, *fndSHA, 1<<20); err != nil {
		return err
	}
	pkg, err := boundRead(*pkgPath, *pkgSHA, wasmrt.LoRADefaultMaxBytes)
	if err != nil {
		return err
	}
	h, _, err := wasmrt.ParseLoRAHeader(pkg)
	if err != nil {
		return err
	}
	if h.Training == nil || h.Training.Family != *intent || len(h.Training.Children) != 1 || h.Training.ChildIntents()[0] != *child {
		return errors.New("selected child training record mismatch")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// The original RequireTraining=false semantics stay. No model-load flag exists.
	art, manifest, err := adapterbuild.CheckLoRA(ctx, adapterbuild.LoRAReq{
		Intent: *intent, PackagePath: *pkgPath, Class: string(c), FoundationPath: *fndPath, OutDir: *out,
	}, nil)
	if err != nil {
		return err
	}
	if art.SHA256 != *pkgSHA || !manifest.Verification.Stage1 || manifest.Verification.Loaded {
		return errors.New("static verification identity mismatch")
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"sha256": art.SHA256, "keccak256": art.Keccak256, "bytes": art.Bytes, "manifest": manifest,
		"class_snapshot_sha256": *snapshotSHA, "foundation_spec_sha256": *fndSHA})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "stagecheck: FAILED:", err)
		os.Exit(1)
	}
}
