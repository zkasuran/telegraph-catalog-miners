// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
package emailsecurity_test

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
)

func TestLiteralBatchInputRoot(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("test source path unavailable")
	}
	source := filepath.Dir(file)
	program := `import sys;sys.path.insert(0,sys.argv[1]);from wire_proxy import input_root;print(input_root([{"tokens":[50281,23,50282]},{"tokens":[50281,42,50282]}]))`
	got, err := exec.Command("python3", "-c", program, source).CombinedOutput()
	if err != nil {
		t.Fatalf("actual packaged Python wire helper: %v %s", err, got)
	}
	want := remote.BatchInputRoot([][]int64{{50281, 23, 50282}, {50281, 42, 50282}})
	if strings.TrimSpace(string(got)) != want {
		t.Fatalf("literal input-root contract mismatch: got %s want %s", got, want)
	}
	if want == remote.BatchInputRoot([][]int64{{50281, 24, 50282}, {50281, 42, 50282}}) {
		t.Fatal("token mutation did not change the real node input root")
	}
}

func TestLiteralInputsQuestion(t *testing.T) {
	f := &responseparser.Format{Inputs: []responseparser.Input{
		{Name: "domain", Description: "Email domain", Type: "string"},
		{Name: "record_type", Description: "DNS record type to look up, e.g. MX or TXT", Type: "string"},
	}}
	in, err := responseparser.DecodeOnChainInputs(f, nil, []string{"_DMARC.Mail.Example.test.", "TXT"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := "EMAIL_SECURITY_GENERAL request. domain (Email domain): _DMARC.Mail.Example.test.. record_type (DNS record type to look up, e.g. MX or TXT): TXT."
	got := responseparser.InputsQuestion("EMAIL_SECURITY_GENERAL", f, in)
	if got != want {
		t.Fatalf("literal wire mismatch: %q", got)
	}
	if got == strings.ToLower(want) {
		t.Fatal("normalized stand-in unexpectedly matched")
	}
	if _, err := responseparser.DecodeOnChainInputs(f, nil, []string{strings.Repeat("x", 257), "TXT"}, nil); err == nil {
		t.Fatal("real upstream size failure did not fire")
	}
}
