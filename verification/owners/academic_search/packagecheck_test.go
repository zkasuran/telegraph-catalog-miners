// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
// Tiny artificial parser fixtures only. No model is loaded or scored here.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

func snapshotFixture() snapshot {
	return snapshot{Schema: 1, FetchedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Family: source{URL: nodeURL + "/api/wasm?resolve=" + familyID, Raw: `{"resolve":{"intent":"ACADEMIC_SEARCH","class":"judgment"}}`},
		Child: source{URL: nodeURL + "/engine/v1/intents/" + childID,
			Raw: `{"intent_id":"ACADEMIC_SEARCH_GENERAL","intent_name":"ACADEMIC_SEARCH_GENERAL","class":"judgment","evaluation":{"kind":"judgment"},"request":null,"response":null}`}}
}

func TestLiveBinding(t *testing.T) {
	s := snapshotFixture()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if cls, err := checkSnapshot(b, familyID, childID); err != nil || cls != "judgment" {
		t.Fatalf("valid live binding: %s %v", cls, err)
	}
	mutations := []snapshot{s, s, s, s}
	mutations[0].FetchedAt = time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano)
	mutations[1].Family.URL += "&untrusted=1"
	mutations[2].Family.Raw = strings.ReplaceAll(s.Family.Raw, "judgment", "hybrid")
	mutations[3].Child.Raw = strings.ReplaceAll(s.Child.Raw, childID, "UNRELATED_GENERAL")
	for i, bad := range mutations {
		b, err := json.Marshal(bad)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := checkSnapshot(b, familyID, childID); err == nil {
			t.Fatalf("mutation %d accepted", i)
		}
	}
}

func tinyPackage(t *testing.T) ([]byte, wasmrt.LoRAExpect) {
	t.Helper()
	spec := wasmrt.FoundationSpec{RMax: 16, HiddenSize: 4, NumLayers: 1, Targets: map[string][2]int{
		"attn.Wqkv": {4, 12}, "attn.Wo": {4, 4}, "mlp.Wi": {4, 16}, "mlp.Wo": {8, 4}}}
	want := wasmrt.LoRAExpect{FoundationHash: strings.Repeat("a", 64), TokenizerHash: strings.Repeat("b", 64), Class: "judgment", Spec: spec}
	names, shapes := wasmrt.LoRAExpectedShapes(8, spec)
	header := map[string]any{"__metadata__": map[string]string{"format_version": "1", "abi": "lora-v1", "class": "judgment",
		"intent": familyID, "foundation_hash": want.FoundationHash, "foundation_version": "toy-parser-only",
		"tokenizer_hash": want.TokenizerHash, "rank": "8", "alpha": "32", "target_modules": `["Wi","Wo","Wqkv"]`, "max_tokens": "256",
		"training": `{"family":"ACADEMIC_SEARCH","children":[{"intent":"ACADEMIC_SEARCH_GENERAL","pairs":64,"accuracy":0.5,"held_out":true}]}`}}
	var offset int
	for _, name := range names {
		size := 4
		for _, dimension := range shapes[name] {
			size *= dimension
		}
		header[name] = map[string]any{"dtype": "F32", "shape": shapes[name], "data_offsets": []int{offset, offset + size}}
		offset += size
	}
	b, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 8+len(b)+offset)
	binary.LittleEndian.PutUint64(raw[:8], uint64(len(b)))
	copy(raw[8:], b)
	return raw, want
}

func TestUnchangedCheckLoRAStaticOnly(t *testing.T) {
	raw, want := tinyPackage(t)
	dir := t.TempDir()
	packagePath := filepath.Join(dir, "fixture.safetensors")
	foundationPath := filepath.Join(dir, "foundation.json")
	fnd := adapterbuild.FoundationFile{FoundationHash: want.FoundationHash, TokenizerHash: want.TokenizerHash,
		FoundationVersion: "toy-parser-only", RMax: want.Spec.RMax, HiddenSize: want.Spec.HiddenSize,
		NumLayers: want.Spec.NumLayers, Targets: want.Spec.Targets}
	encoded, err := json.Marshal(fnd)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(foundationPath, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(packagePath, raw, 0600); err != nil {
		t.Fatal(err)
	}
	catalog.UseClasses(map[string]catalog.Class{familyID: catalog.ClassJudgment, childID: catalog.ClassJudgment})
	defer catalog.ClearClasses()
	_, man, err := adapterbuild.CheckLoRA(context.Background(), adapterbuild.LoRAReq{Intent: familyID,
		Class: "judgment", PackagePath: packagePath, FoundationPath: foundationPath, OutDir: dir}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !man.Verification.Stage1 || man.Verification.Loaded {
		t.Fatal("full checker skipped Stage 1 or loaded a model")
	}
}

func TestFullUpstreamTensorParser(t *testing.T) {
	raw, want := tinyPackage(t)
	if _, err := wasmrt.ParseLoRAAdapter(raw, want); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"foundation", "tokenizer", "class", "nonfinite", "truncated", "trailing"} {
		t.Run(kind, func(t *testing.T) {
			bad := append([]byte(nil), raw...)
			expected := want
			switch kind {
			case "foundation":
				expected.FoundationHash = strings.Repeat("c", 64)
			case "tokenizer":
				expected.TokenizerHash = strings.Repeat("c", 64)
			case "class":
				expected.Class = "hybrid"
			case "nonfinite":
				binary.LittleEndian.PutUint32(bad[len(bad)-4:], 0x7fc00000)
			case "truncated":
				bad = bad[:len(bad)-1]
			case "trailing":
				bad = append(bad, 0)
			}
			if _, err := wasmrt.ParseLoRAAdapter(bad, expected); err == nil {
				t.Fatal("full parser accepted mutation")
			}
		})
	}
}
