package comparator

import (
    "bytes"
    "context"
    "crypto/sha256"
    "encoding/binary"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "io"
    "math"
    "math/rand"
    "net/http"
    "os"
    "testing"

    "github.com/tetratelabs/wazero"
    "github.com/tetratelabs/wazero/api"
    "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
    wr "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

type rawComparator struct { mod api.Module; ctx context.Context }

func openRaw(t *testing.T, b []byte) rawComparator {
    t.Helper()
    ctx := context.Background()
    r := wazero.NewRuntime(ctx)
    t.Cleanup(func() { if err := r.Close(ctx); err != nil { t.Error(err) } })
    mod, err := r.Instantiate(ctx, b)
    if err != nil { t.Fatal(err) }
    return rawComparator{mod, ctx}
}

func (r rawComparator) score(t *testing.T, truth, miner int64, spec []byte) int64 {
    t.Helper()
    a, err := r.mod.ExportedFunction("imw_alloc").Call(r.ctx, 16)
    if err != nil || len(a) != 1 || a[0] == 0 { t.Fatalf("allocation %v %v", a, err) }
    p := uint32(a[0])
    if !r.mod.Memory().Write(p, spec) { t.Fatal("spec write failed") }
    got, err := r.mod.ExportedFunction("imw_score_deterministic").Call(r.ctx, uint64(truth), uint64(miner), uint64(p))
    if err != nil || len(got) != 1 { t.Fatalf("raw score %v %v", got, err) }
    if _, err := r.mod.ExportedFunction("imw_free").Call(r.ctx, uint64(p), 16); err != nil { t.Fatal(err) }
    return int64(got[0])
}

func specBytes(code, tol int64) []byte {
    b := make([]byte, 16)
    binary.LittleEndian.PutUint64(b[:8], uint64(code))
    binary.LittleEndian.PutUint64(b[8:], uint64(tol))
    return b
}

func comparatorBytes(t *testing.T) []byte {
    t.Helper()
    p := os.Getenv("ZK_COMPARATOR_WASM")
    if p == "" { t.Fatal("ZK_COMPARATOR_WASM is mandatory") }
    b, err := os.ReadFile(p)
    if err != nil { t.Fatal(err) }
    return b
}

func TestIndependentRawBoundaries(t *testing.T) {
    b := comparatorBytes(t)
    raw := openRaw(t, b)
    ctx := context.Background()
    host, err := wr.NewDeterministic(ctx, b)
    if err != nil { t.Fatal(err) }
    t.Cleanup(func() { if err := host.Close(ctx); err != nil { t.Error(err) } })
    // Goldens are explicit independent expectations, not formulas copied from the implementation.
    goldens := []struct { name string; code, tol, truth, miner, want int64 }{
        {"exact above 2^53 equality", 0, 0, 9007199254740993, 9007199254740993, 10000},
        {"adjacent above 2^53 distinction", 0, 0, 9007199254740993, 9007199254740992, 0},
        {"truth-first denominator", 2, 50, 10000, 10099, 100},
        {"swapped denominator changes score", 2, 50, 10099, 10000, 200},
        {"negative temperature half band", 1, 20, -100, -80, 5000},
        {"negative temperature window edge", 1, 20, -100, -60, 0},
        {"negative temperature just inside", 1, 20, -100, -61, 250},
        {"negative temperature just outside", 1, 20, -100, -59, 0},
        {"negative tolerance equal", 1, -1, 7, 7, 10000},
        {"zero tolerance unequal", 2, 0, 10000, 10001, 0},
        {"zero tolerance sub-basis-point error", 2, 0, 100000, 100001, 10000},
        {"zero truth raw PassFail equality", 3, 0, 0, 0, 10000},
        {"MinInt64 exact equality", 0, 0, math.MinInt64, math.MinInt64, 10000},
        {"MinInt64 abs one unit", 1, 1, math.MinInt64, math.MinInt64 + 1, 5000},
        {"MinInt64 rel denominator becomes one", 2, 50, math.MinInt64, math.MinInt64 + 1, 0},
        {"negative quotient below int64 clamps MAX", 2, 50, 1, math.MinInt64 + 1, 0},
        {"positive quotient above int64 clamps MAX", 2, 50, 1, math.MaxInt64, 0},
        {"wrapping difference can score half", 1, 1, math.MaxInt64, math.MinInt64, 5000},
        {"wrapping difference can score perfect", 2, 50, math.MaxInt64, math.MinInt64, 10000},
        {"wrapped negative raw refusal", 1, 1, math.MaxInt64, -1, -4611686018427387904},
        {"large tol saturated numerator equality", 1, math.MaxInt64, 1, 1, 1},
        {"large tol RelTol equality", 2, math.MaxInt64, 1, 1, 1},
        {"truncated i64 code becomes RelTol", 1<<32 + 2, 50, 10000, 10099, 100},
        {"unknown code Exact", 4, 100, 10000, 10099, 0},
        {"int64 Max code becomes unknown", math.MaxInt64, 50, 10000, 10099, 0},
    }
    n, positive, zero, outside := 0, 0, 0, 0
    check := func(code, tol, truth, miner int64) int64 {
        want := ReferenceScore(code, tol, truth, miner)
        got := raw.score(t, truth, miner, specBytes(code, tol))
        n++
        if got != want { t.Fatalf("raw (%d,%d,%d,%d): got %d want %d", code, tol, truth, miner, got, want) }
        switch { case got < 0 || got > 10000: outside++; case got == 0: zero++; default: positive++ }
        wrapped, e := host.ScoreDeterministic(ctx, miner, truth, [2]int64{code, tol})
        if truth == 0 || want < 0 || want > 10000 {
            if e == nil { t.Fatalf("expected refusal, got %d", wrapped) }
        } else if e != nil || int64(wrapped) != want { t.Fatalf("host: %d %v want %d", wrapped, e, want) }
        return got
    }
    for _, g := range goldens {
        got := check(g.code, g.tol, g.truth, g.miner)
        if got != g.want { t.Fatalf("%s: %d want %d", g.name, got, g.want) }
        t.Logf("golden %s: code=%d tol=%d truth=%d miner=%d raw=%d", g.name, g.code, g.tol, g.truth, g.miner, got)
    }
    values := []int64{math.MinInt64, math.MinInt64+1, math.MaxInt64, math.MaxInt64-1, -9007199254740993, -100, -1, 0, 1, 100, 9007199254740993}
    for _, c := range []int64{0,1,2,3,4,-1,1<<32+1,1<<32+2} {
        for _, tol := range []int64{math.MinInt64,-1,0,1,20,50,math.MaxInt64/2,math.MaxInt64} {
            for _, truth := range values { for _, miner := range values { check(c,tol,truth,miner) } }
        }
    }
    rng := rand.New(rand.NewSource(2026111801))
    for i:=0; i<20000; i++ { check(int64(rng.Intn(5)), int64(rng.Uint64()), int64(rng.Uint64()), int64(rng.Uint64())) }
    if positive == 0 || zero == 0 || outside == 0 { t.Fatalf("vacuous coverage: %d %d %d",positive,zero,outside) }
    t.Logf("raw coverage: %d cases, %d positive, %d zero, %d outside domain, zero mismatches", n,positive,zero,outside)
    // Repeat actual nonzero results and assert errors as well as score bits.
    repeated := 0
    for i:=0; i<500; i++ {
        truth := int64(10000+i)
        for _, m := range []int64{truth,truth+1,truth+25,truth+50,truth+75} {
            want := ReferenceScore(2,50,truth,m)
            if want <= 0 { t.Fatalf("repeatability fixture must score positive: %d",want) }
            for j:=0;j<2;j++ { got,e:=host.ScoreDeterministic(ctx,m,truth,[2]int64{2,50}); if e!=nil || int64(got)!=want { t.Fatalf("repeat: %d %v want %d",got,e,want) } }
            repeated++
        }
    }
    t.Logf("repeatability: %d positive cases repeated twice without ignored errors", repeated)
    sum:=sha256.Sum256(b)
    t.Logf("checked exact WASM SHA256 %x bytes %d",sum,len(b))
}

func TestIndependentByteOrderControls(t *testing.T) {
    raw := openRaw(t,comparatorBytes(t))
    le:=specBytes(1<<32+2,50)
    if hex.EncodeToString(le)!="02000000010000003200000000000000" { t.Fatal("byte fixture changed") }
    if got:=raw.score(t,10000,10099,le);got!=100 { t.Fatalf("LE got %d",got) }
    be:=make([]byte,16)
    binary.BigEndian.PutUint64(be[:8],1<<32+2);binary.BigEndian.PutUint64(be[8:],50)
    if got:=raw.score(t,10000,10099,be);got!=0 { t.Fatalf("BE control got %d",got) }
    if got:=raw.score(t,10000,10099,specBytes(50,2));got!=0 { t.Fatalf("field swap got %d",got) }
    if got:=raw.score(t,10099,10000,le);got!=200 { t.Fatalf("argument swap got %d",got) }
    t.Log("byte controls: LE 02000000010000003200000000000000 scores 100; BE 0; swapped fields 0; swapped values 200")
    if got,e:=raw.mod.ExportedFunction("imw_alloc").Call(raw.ctx,4097);e!=nil || len(got)!=1 || got[0]!=0 { t.Fatalf("oversize alloc: %v %v",got,e) }
    p:=raw.mod.Memory().Size()-8
    if _,e:=raw.mod.ExportedFunction("imw_score_deterministic").Call(raw.ctx,1,1,uint64(p));e==nil { t.Fatal("invalid 16-byte spec must trap") }
    t.Log("allocator 4097 refused; spec crossing memory end trapped")
}

type captureTransport struct { body []byte }
func (c *captureTransport) RoundTrip(r *http.Request) (*http.Response,error) {
    b,e:=io.ReadAll(r.Body); if e!=nil{return nil,e};c.body=b
    if r.Method!="POST" || r.Header.Get("Content-Type")!="application/json" { return nil,fmt.Errorf("wrong caller method or content type") }
    return &http.Response{StatusCode:200,Body:io.NopCloser(bytes.NewBufferString(`{"results":[{"score":10000},{"score":5000}]}`)),Header:make(http.Header)},nil
}
func TestIndependentCallerWire(t *testing.T) {
    capture:=&captureTransport{}
    caller:=&remote.RemoteExecutor{URL:"https://unused.invalid/runnable",Client:&http.Client{Transport:capture}}
    out,e:=caller.ScoreDeterministicBatch(context.Background(),[]remote.DeterministicCase{
        {MinerValue:9007199254740992,TruthValue:9007199254740993,Spec:[2]int64{2,50}},
        {MinerValue:math.MinInt64+1,TruthValue:math.MinInt64,Spec:[2]int64{1,20}},
    })
    if e!=nil || len(out)!=2 || out[0].Score!=10000 || out[1].Score!=5000 {t.Fatalf("caller result %v %v",out,e)}
    var body struct {Batch []struct {Truth int64 `json:"ground_truth_value"`;Miner int64 `json:"miner_value"`;Spec [2]int64 `json:"spec"`}}
    if e:=json.Unmarshal(capture.body,&body);e!=nil {t.Fatal(e)}
    if len(body.Batch)!=2 || body.Batch[0].Truth!=9007199254740993 || body.Batch[0].Miner!=9007199254740992 || body.Batch[1].Truth!=math.MinInt64 || body.Batch[0].Spec!=[2]int64{2,50} {t.Fatalf("wire precision/order failed: %s",capture.body)}
    t.Logf("actual caller wire captured in memory: %s",capture.body)
    t.Log("no socket or external request was made; in-memory RoundTripper only")
}
