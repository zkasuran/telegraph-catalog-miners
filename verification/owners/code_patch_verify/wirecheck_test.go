// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
package wirecheck

import (
	"context"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/comparator"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

func TestLiteralRequestBuilder(t *testing.T) {
	url := "https://github.com/cpv-fixture/source/actions/runs/900000/attempts/1"
	format := &responseparser.Format{Inputs: []responseparser.Input{{Name: "patch", Description: "Repository and commit, pull request or CI run to check", Type: responseparser.TypeString}}}
	got := responseparser.InputsQuestion("CODE_PATCH_VERIFY_GENERAL", format, responseparser.OrderedObject{{Key: "patch", Value: url}})
	want := "CODE_PATCH_VERIFY_GENERAL request. patch (Repository and commit, pull request or CI run to check): " + url + "."
	if got != want {
		t.Fatalf("literal request mismatch: got %q", got)
	}
}

func TestDefaultRawAdapterInputPreservesLiteralBytes(t *testing.T) {
	for _, body := range []string{" \t{\"passed\":true,\"conclusion\":\"success\"}\r\n", "{\"conclusion\":\"failure\",\"passed\":false,\"summary\":\"é\"}"} {
		got := comparator.AdapterInput(context.Background(), "CODE_PATCH_VERIFY_GENERAL", "source-only-check", "a question", body)
		if got != body {
			t.Fatal("reference default changed literal candidate bytes")
		}
	}
}

func TestRealOnChainBooleanLayoutAndNegativeControls(t *testing.T) {
	format := &responseparser.Format{Fields: []responseparser.Field{
		{Name: "conclusion", Type: responseparser.TypeEnum},
		{Name: "passed", Type: responseparser.TypeBool},
		{Name: "commit", Type: responseparser.TypeString, OffChain: true},
		{Name: "pipeline", Type: responseparser.TypeString, OffChain: true},
	}}
	makeAnswer := func(value any) responseparser.Shaped {
		return responseparser.Shaped{Fields: []responseparser.FieldValue{
			{Status: responseparser.FieldOK, Value: "success"},
			{Status: responseparser.FieldOK, Value: value},
			{Status: responseparser.FieldOK, Value: "commit"},
			{Status: responseparser.FieldOK, Value: "pipeline"},
		}}
	}
	for _, value := range []bool{true, false} {
		got, err := responseparser.EncodeOnChain(format, makeAnswer(value))
		if err != nil || len(got.Bools) != 1 || got.Bools[0] != value || len(got.Strings) != 1 {
			t.Fatalf("native boolean/layout mismatch: %v", err)
		}
	}
	for _, value := range []any{"true", float64(1), nil, []string{"true"}} {
		if _, err := responseparser.EncodeOnChain(format, makeAnswer(value)); err == nil {
			t.Fatal("reference accepted a non-native boolean")
		}
	}
	missing := makeAnswer(true)
	missing.Fields[1].Status = "unavailable"
	if _, err := responseparser.EncodeOnChain(format, missing); err == nil {
		t.Fatal("missing value was silently encoded")
	}
}
