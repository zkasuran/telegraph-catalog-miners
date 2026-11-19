// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
package main

// These tests call the pinned counterparty handler and literal node client.
// The fake scorer is a wire fixture, not a neural model or a quality measure.
import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

type ownerFakeScorer struct{}

func (ownerFakeScorer) Identity() wasmrt.ScoreIdentity {
	return wasmrt.ScoreIdentity{FoundationHash: "aaaabb", AdapterHash: "ccddee", TokenizerHash: "112233"}
}
func (ownerFakeScorer) ScoreTokens(_ context.Context, ids []int64) (wasmrt.ModelScore, error) {
	if len(ids) == 0 {
		return wasmrt.ModelScore{}, errors.New("empty fixture tokens")
	}
	return wasmrt.NewModelScoreFromBits(0x3f800000)
}

func TestOwnerLiteralWire(t *testing.T) {
	srv := httptest.NewServer(handler(ownerFakeScorer{}))
	defer srv.Close()
	client := &remote.RemoteExecutor{URL: srv.URL}
	batch := [][]int64{{1, 2}, {1, 3, 2}, {1, 2}}
	ident := remote.BatchIdentity{FoundationHash: "aaaabb", AdapterHash: "ccddee", TokenizerHash: "112233", InputRoot: remote.BatchInputRoot(batch)}
	res, echoed, err := client.ScoreBatchWithIdentity(context.Background(), batch, &ident)
	if err != nil || !echoed || len(res) != 3 || res[0].Score.Bits() != res[2].Score.Bits() {
		t.Fatalf("primary literal node-client path failed: %v %+v %v", err, res, echoed)
	}
	ident.AdapterHash = "0x" + ident.AdapterHash
	_, _, err = client.ScoreBatchWithIdentity(context.Background(), batch, &ident)
	var rejected *remote.RemoteRejectedError
	if !errors.As(err, &rejected) {
		t.Fatalf("literal 0x mismatch was normalized away: %v", err)
	}
	resp, err := http.Post(srv.URL, "application/json", bytes.NewReader([]byte(`{"tokens":[1,2]}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("missing batch status %d", resp.StatusCode)
	}
}
