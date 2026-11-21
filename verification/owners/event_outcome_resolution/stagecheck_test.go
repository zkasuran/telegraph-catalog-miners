// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func sourceFixture(t *testing.T) snapshot {
	t.Helper()
	return snapshot{Schema: 1, FetchedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Family: source{URL: "https://devnode.telegraphprotocol.com/api/wasm?resolve=SENTIMENT_ANALYSIS",
			Raw: `{"resolve":{"intent":"SENTIMENT_ANALYSIS","class":"judgment","served_by":"GENERIC"}}`},
		Child: source{URL: "https://devnode.telegraphprotocol.com/engine/v1/intents/SENTIMENT_ANALYSIS_GENERAL",
			Raw: `{"intent_id":"SENTIMENT_ANALYSIS_GENERAL","intent_name":"SENTIMENT_ANALYSIS_GENERAL","class":"judgment","evaluation":{"kind":"judgment"},"request":{"served_by":"SENTIMENT_ANALYSIS"},"response":{"served_by":"SENTIMENT_ANALYSIS"},"onchain":{"served_by":"SENTIMENT_ANALYSIS","ready":true}}`}}
}

func encodedFixture(t *testing.T, value snapshot) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestCheckedClassSourceAcceptsFamilyFallback(t *testing.T) {
	c, err := checkSource(encodedFixture(t, sourceFixture(t)), "SENTIMENT_ANALYSIS", "SENTIMENT_ANALYSIS_GENERAL", "judgment")
	if err != nil || c != catalog.ClassJudgment {
		t.Fatalf("checked source failed: %v", err)
	}
}

func TestClassIdentityTimeAndOverrideRefusals(t *testing.T) {
	base := sourceFixture(t)
	mutations := []snapshot{}
	wrong := base
	wrong.Child.Raw = strings.Replace(base.Child.Raw, `"class":"judgment"`, `"class":"hybrid"`, 1)
	mutations = append(mutations, wrong)
	wrong = base
	wrong.FetchedAt = time.Now().Add(-16 * time.Minute).UTC().Format(time.RFC3339)
	mutations = append(mutations, wrong)
	wrong = base
	wrong.Family.URL = "https://example.com"
	mutations = append(mutations, wrong)
	wrong = base
	wrong.Child.Raw = strings.ReplaceAll(base.Child.Raw, `"served_by":"SENTIMENT_ANALYSIS"`, `"served_by":"GENERIC"`)
	mutations = append(mutations, wrong)
	for _, value := range mutations {
		if _, err := checkSource(encodedFixture(t, value), "SENTIMENT_ANALYSIS", "SENTIMENT_ANALYSIS_GENERAL", "judgment"); err == nil {
			t.Fatal("invalid source accepted")
		}
	}
	if _, err := checkSource(encodedFixture(t, base), "SENTIMENT_ANALYSIS", "SENTIMENT_ANALYSIS_GENERAL", "hybrid"); err == nil {
		t.Fatal("class override accepted")
	}
}

func parentFixture(t *testing.T) snapshot {
	t.Helper()
	return snapshot{Schema: 1, FetchedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Family: source{URL: "https://devnode.telegraphprotocol.com/api/wasm?resolve=EVENT_OUTCOME_RESOLUTION",
			Raw: `{"resolve":{"intent":"EVENT_OUTCOME_RESOLUTION","class":"judgment"}}`},
		Child: source{URL: "https://devnode.telegraphprotocol.com/engine/v1/intents/EVENT_OUTCOME_RESOLUTION",
			Raw: `{"intent_id":"EVENT_OUTCOME_RESOLUTION","intent_name":"EVENT_OUTCOME_RESOLUTION","class":"judgment","prefix":true,"request":{"served_by":"EVENT_OUTCOME_RESOLUTION"},"response":{"served_by":"EVENT_OUTCOME_RESOLUTION"},"onchain":{"served_by":"EVENT_OUTCOME_RESOLUTION","ready":true,"format_version":4,"request":[{"name":"event","type":"string","slot":"strings","index":0,"max_bytes":256}],"response":[{"name":"outcome","type":"enum","slot":"strings","index":0,"allowed":["yes","no","unresolved"],"max_bytes":256},{"name":"resolved_yes","type":"bool","slot":"bools","index":0}]}}`}}
}

func TestParentOnlyScopedParserAcceptsExactV4Source(t *testing.T) {
	value := parentFixture(t)
	c, err := checkScopedSource(encodedFixture(t, value), "EVENT_OUTCOME_RESOLUTION", "EVENT_OUTCOME_RESOLUTION", "judgment", "parent_only")
	if err != nil || c != catalog.ClassJudgment {
		t.Fatalf("parent-only scope rejected: %v", err)
	}
	if _, err := checkSource(encodedFixture(t, value), "EVENT_OUTCOME_RESOLUTION", "EVENT_OUTCOME_RESOLUTION", "judgment"); err == nil {
		t.Fatal("general scope silently accepted parent-only source")
	}
}

func TestParentOnlyScopeNeverOverridesClassOrWire(t *testing.T) {
	base := parentFixture(t)
	for _, replacement := range [][2]string{{`"prefix":true`, `"prefix":false`}, {`"format_version":4`, `"format_version":1`},
		{`"max_bytes":256`, `"max_bytes":512`}, {`"class":"judgment"`, `"class":"hybrid"`}, {`"type":"bool"`, `"type":"enum"`}} {
		value := base
		value.Child.Raw = strings.Replace(base.Child.Raw, replacement[0], replacement[1], 1)
		if _, err := checkScopedSource(encodedFixture(t, value), "EVENT_OUTCOME_RESOLUTION", "EVENT_OUTCOME_RESOLUTION", "judgment", "parent_only"); err == nil {
			t.Fatal("invalid parent source accepted")
		}
	}
}

func TestExternalPackageHashAndSizeGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source-fixture")
	raw := []byte("not a model")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(raw)
	expected := hex.EncodeToString(h[:])
	if _, err := boundRead(path, expected, 1024); err != nil {
		t.Fatal(err)
	}
	if _, err := boundRead(path, strings.Repeat("0", 64), 1024); err == nil {
		t.Fatal("wrong file hash accepted")
	}
	if _, err := boundRead(path, expected, 1); err == nil {
		t.Fatal("oversized input accepted")
	}
}

func TestRealCheckLoRARejectsInvalidContainerWithoutLoad(t *testing.T) {
	root := t.TempDir()
	fnd := filepath.Join(root, "foundation.json")
	data := `{"foundation_hash":"0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6","tokenizer_hash":"67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456","foundation_version":"modernbert-base-lora16-v1","r_max":16,"num_layers":22,"hidden_size":768,"targets":{"attn.Wqkv":[768,2304],"attn.Wo":[768,768],"mlp.Wi":[768,2304],"mlp.Wo":[1152,768]}}`
	if err := os.WriteFile(fnd, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	pkg := filepath.Join(root, "invalid.safetensors")
	if err := os.WriteFile(pkg, []byte("invalid source-only container"), 0600); err != nil {
		t.Fatal(err)
	}
	catalog.UseClasses(map[string]catalog.Class{"SENTIMENT_ANALYSIS": catalog.ClassJudgment, "SENTIMENT_ANALYSIS_GENERAL": catalog.ClassJudgment})
	defer catalog.ClearClasses()
	_, manifest, err := adapterbuild.CheckLoRA(context.Background(), adapterbuild.LoRAReq{
		Intent: "SENTIMENT_ANALYSIS", Class: "judgment", PackagePath: pkg, FoundationPath: fnd, OutDir: root,
	}, nil)
	if err == nil || manifest.Verification.Loaded {
		t.Fatal("invalid static package accepted or model loaded")
	}
}

func TestNodeInputRootMatchesIndependentGolden(t *testing.T) {
	ids := [][]int64{{1, 2}, {1, 2}, {3}}
	h1 := sha256.Sum256([]byte(`{"input_ids":[1,2],"attention_mask":[1,1]}`))
	h2 := sha256.Sum256([]byte(`{"input_ids":[3],"attention_mask":[1]}`))
	text := "0:" + hex.EncodeToString(h1[:]) + "\n1:" + hex.EncodeToString(h1[:]) + "\n2:" + hex.EncodeToString(h2[:])
	root := sha256.Sum256([]byte(text))
	if remote.BatchInputRoot(ids) != hex.EncodeToString(root[:]) {
		t.Fatal("node canonical input encoding differs from proxy contract")
	}
}
