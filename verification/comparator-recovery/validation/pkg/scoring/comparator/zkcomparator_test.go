package comparator

import (
	"context"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// ZK_COMPARATOR_WASM=/path/to/zk_comparator.wasm go test ./pkg/scoring/comparator -run TestZKComparator -v
func TestZKComparator(t *testing.T) {
	path := os.Getenv("ZK_COMPARATOR_WASM")
	if path == "" {
		t.Skip("ZK_COMPARATOR_WASM not set")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	rt, err := runtime.NewDeterministic(ctx, b)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close(ctx)

	truths := []int64{1, 2, 3, 7, 10, 99, 100, 101, 1000, 12345, 1_000_000, 1 << 31, 1<<31 + 1, 1 << 40, 1e15, 1e18, 9e18,
		math.MaxInt64, -1, -7, -100, -12345, -1e15, -1e18, -9e18, math.MinInt64 + 1}
	codes := []int64{0, 1, 2, 3, 4, 5, -1, 255, 1 << 32, 1<<32 + 1, 1<<32 + 2, 1<<33 + 3, math.MaxInt64, math.MinInt64}
	tols := []int64{math.MinInt64, -1000, -1, 0, 1, 2, 3, 5, 10, 50, 100, 1000, 10000, 1_000_000, 1e12, 1e18, math.MaxInt64 / 2, math.MaxInt64}
	r := rand.New(rand.NewSource(20261103))
	n, bad, refused := 0, 0, 0
	check := func(code, tol, truth, miner int64) {
		n++
		want := ReferenceScore(code, tol, truth, miner)
		got, err := rt.ScoreDeterministic(ctx, miner, truth, [2]int64{code, tol})
		if want < 0 || want > ReferenceScoreOne {
			// The reference itself leaves 0..10000 only when miner-truth overflows int64. The node treats such a return as
			// "the module refused the input"; the module must refuse the same inputs.
			refused++
			if err == nil {
				bad++
				t.Errorf("code=%d tol=%d truth=%d miner=%d: reference=%d is out of range but the module scored %d", code, tol, truth, miner, want, got)
			}
			return
		}
		if err != nil || int64(got) != want {
			bad++
			if bad <= 12 {
				t.Errorf("code=%d tol=%d truth=%d miner=%d: module=%d err=%v, reference=%d", code, tol, truth, miner, got, err, want)
			}
		}
	}
	for _, code := range codes {
		for _, tol := range tols {
			for _, truth := range truths {
				cands := []int64{truth, truth + 1, truth - 1, 0, -truth, truth * 2, truth / 2, math.MaxInt64, math.MinInt64}
				// half-score point, window edge and just past it, on both sides (AbsTol: err = tol, 2*tol; RelTol: truth*tol/10000)
				for _, d := range []int64{tol, 2 * tol, 2*tol - 1, 2*tol + 1, tol / 2, truth / 10000 * tol / 2} {
					cands = append(cands, truth+d, truth-d)
				}
				for _, m := range cands {
					check(code, tol, truth, m)
				}
			}
		}
	}
	for i := 0; i < 400000; i++ { // random, including wraparound-prone magnitudes
		var truth int64
		for truth == 0 {
			truth = int64(r.Uint64()) >> uint(r.Intn(63))
		}
		check(codes[r.Intn(len(codes))], int64(r.Uint64())>>uint(r.Intn(63)), truth, int64(r.Uint64())>>uint(r.Intn(63)))
	}
	t.Logf("%d cases, %d mismatches (%d overflow inputs refused by both)", n, bad, refused)
	if bad > 0 {
		t.Fatalf("%d of %d cases differ from the reference", bad, n)
	}
	// determinism: a second identical run must return exactly the same
	for i := 0; i < 2000; i++ {
		truth, miner := int64(r.Intn(1e9)+1), int64(r.Intn(1e9))
		a, _ := rt.ScoreDeterministic(ctx, miner, truth, [2]int64{2, 100})
		c, _ := rt.ScoreDeterministic(ctx, miner, truth, [2]int64{2, 100})
		if a != c {
			t.Fatalf("not deterministic: %d vs %d", a, c)
		}
	}
}
