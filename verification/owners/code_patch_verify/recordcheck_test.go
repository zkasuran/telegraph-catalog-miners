// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Synthetic source contract controls, not a model or independent factual audit.
package recordcheck

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

func controlRecords(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile(os.Getenv("CPV_SOURCE_RECORD_CONTROLS"))
	if err != nil || len(raw) > 65536 {
		t.Fatalf("bounded synthetic source control missing: %v", err)
	}
	var records map[string]json.RawMessage
	if err := json.Unmarshal(raw, &records); err != nil {
		t.Fatal(err)
	}
	return records
}

func TestOriginalMalformedTrainingRecordRefused(t *testing.T) {
	var record wasmrt.TrainingRecord
	if err := json.Unmarshal(controlRecords(t)["original"], &record); err != nil {
		t.Fatal(err)
	}
	if err := record.Validate(); err == nil || !strings.Contains(err.Error(), "children is empty") {
		t.Fatalf("original source regression was not refused: %v", err)
	}
}

func TestActualNodeParserAcceptsMeasuredRecordWire(t *testing.T) {
	var record wasmrt.TrainingRecord
	if err := json.Unmarshal(controlRecords(t)["fixed"], &record); err != nil {
		t.Fatal(err)
	}
	if err := record.Validate(); err != nil {
		t.Fatal(err)
	}
	if record.Family != "CODE_PATCH_VERIFY" || len(record.Children) != 1 || record.ChildIntents()[0] != "CODE_PATCH_VERIFY_GENERAL" {
		t.Fatal("selected child wire identity changed")
	}
	meta := wasmrt.AdapterMeta{ABIVersion: 2, Class: "judgment", Runtime: "lora-v1", MaxTokens: 512, Training: &record}
	if err := meta.Check("judgment"); err != nil {
		t.Fatal(err)
	}
	meta.ABIVersion = 1
	if err := meta.Check("judgment"); err == nil {
		t.Fatal("retired model ABI was accepted")
	}
}

func TestActualNodeParserRejectsForeignChild(t *testing.T) {
	var record wasmrt.TrainingRecord
	if err := json.Unmarshal(controlRecords(t)["fixed"], &record); err != nil {
		t.Fatal(err)
	}
	record.Children[0].Intent = "OTHER_GENERAL"
	if err := record.Validate(); err == nil {
		t.Fatal("foreign child was accepted")
	}
}
