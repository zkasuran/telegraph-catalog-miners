// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
// Product-report scope wrapper. Every tensor/header check stays in the pinned upstream parser.
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

const familyID = "PRODUCT_AUTHENTICITY"
const childID = familyID + "_GENERAL"
const nodeURL = "https://devnode.telegraphprotocol.com"

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
	if path == "" || !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(expected) {
		return nil, errors.New("input path and external lowercase SHA-256 required")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return nil, errors.New("input must be a bounded regular file")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, errors.New("input byte ceiling")
	}
	h := sha256.Sum256(b)
	if hex.EncodeToString(h[:]) != expected {
		return nil, errors.New("input SHA-256 mismatch")
	}
	return b, nil
}

func checkSnapshot(raw []byte, intent, child string) (catalog.Class, error) {
	if intent != familyID || child != childID {
		return "", errors.New("exact product family family and child required")
	}
	var s snapshot
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return "", err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return "", errors.New("one snapshot JSON object required")
	}
	if s.Schema != 1 || s.Family.URL != nodeURL+"/api/wasm?resolve="+intent || s.Child.URL != nodeURL+"/engine/v1/intents/"+child {
		return "", errors.New("authoritative source URL mismatch")
	}
	t, err := time.Parse(time.RFC3339Nano, s.FetchedAt)
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
		IntentID   string        `json:"intent_id"`
		IntentName string        `json:"intent_name"`
		Class      catalog.Class `json:"class"`
		Evaluation struct {
			Kind string `json:"kind"`
		} `json:"evaluation"`
		Request struct {
			ServedBy string `json:"served_by"`
		} `json:"request"`
		Response struct {
			ServedBy string `json:"served_by"`
		} `json:"response"`
		Onchain struct {
			ServedBy string `json:"served_by"`
		} `json:"onchain"`
	}
	if err := json.Unmarshal([]byte(s.Family.Raw), &parent); err != nil {
		return "", err
	}
	if err := json.Unmarshal([]byte(s.Child.Raw), &descendant); err != nil {
		return "", err
	}
	c := parent.Resolve.Class
	if c != catalog.ClassJudgment || parent.Resolve.Intent != intent || descendant.IntentID != child ||
		descendant.IntentName != child || descendant.Class != c || descendant.Evaluation.Kind != string(c) {
		return "", errors.New("live product family class and identity agreement failed")
	}
	for _, owner := range []string{descendant.Request.ServedBy, descendant.Response.ServedBy, descendant.Onchain.ServedBy} {
		if owner != "" && owner != intent {
			return "", errors.New("advertised schema belongs to another family")
		}
	}
	return c, nil
}

func run() error {
	intent := flag.String("intent", "", "exact family")
	child := flag.String("child", "", "exact child")
	snapshotPath := flag.String("class-snapshot", "", "trusted HTTPS snapshot")
	snapshotSHA := flag.String("snapshot-sha256", "", "external snapshot SHA-256")
	packagePath := flag.String("package", "", "unchanged package")
	packageSHA := flag.String("sha256", "", "external package SHA-256")
	foundationPath := flag.String("foundation", "", "pinned foundation layout")
	foundationSHA := flag.String("foundation-sha256", "", "external foundation JSON SHA-256")
	out := flag.String("out", "", "output directory")
	preflight := flag.Bool("preflight", false, "class only, no package or model")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected positional arguments")
	}
	s, err := boundRead(*snapshotPath, *snapshotSHA, 2<<20)
	if err != nil {
		return fmt.Errorf("snapshot: %w", err)
	}
	c, err := checkSnapshot(s, *intent, *child)
	if err != nil {
		return fmt.Errorf("snapshot: %w", err)
	}
	// UseClasses is upstream's public tool hook. Only checked primary identities enter it.
	catalog.UseClasses(map[string]catalog.Class{*intent: c, *child: c})
	defer catalog.ClearClasses()
	resolved, err := adapterbuild.ResolveClass(*intent, "")
	if err != nil {
		return err
	}
	if *preflight {
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"intent": *intent, "child": *child, "class": resolved,
			"snapshot_sha256": *snapshotSHA, "preflight": true, "loaded": false})
	}
	if *out == "" {
		return errors.New("output directory required")
	}
	if _, err := boundRead(*foundationPath, *foundationSHA, 1<<20); err != nil {
		return fmt.Errorf("foundation: %w", err)
	}
	pkg, err := boundRead(*packagePath, *packageSHA, wasmrt.LoRADefaultMaxBytes)
	if err != nil {
		return fmt.Errorf("package: %w", err)
	}
	h, _, err := wasmrt.ParseLoRAHeader(pkg)
	if err != nil {
		return err
	}
	if h.Training == nil || h.Training.Family != *intent || len(h.Training.Children) != 1 || h.Training.ChildIntents()[0] != *child {
		return errors.New("training record must name the selected QA family and child")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// No model/engine paths. This executes every real Stage 1 header/tensor check.
	art, man, err := adapterbuild.CheckLoRA(ctx, adapterbuild.LoRAReq{Intent: *intent, PackagePath: *packagePath,
		Class: resolved, FoundationPath: *foundationPath, OutDir: *out}, nil)
	if err != nil {
		return err
	}
	if art.SHA256 != *packageSHA || man.Verification.Loaded || !man.Verification.Stage1 || man.Rank != 8 || man.MaxTokens != 256 {
		return errors.New("package binding, Stage 1 or fixed experiment layout failed")
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"intent": *intent, "class": man.Class, "sha256": art.SHA256,
		"keccak256": art.Keccak256, "bytes": art.Bytes, "manifest": man,
		"snapshot_sha256": *snapshotSHA, "foundation_spec_sha256": *foundationSHA})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "qa-packagecheck: FAILED:", err)
		os.Exit(1)
	}
}
