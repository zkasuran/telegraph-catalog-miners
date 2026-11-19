// SPDX-License-Identifier: MIT
// Copyright (c) 2026 telegraphprotocol
// Unchanged helper excerpts needed to compile remote.go. No model executor is included.
package runtime
import ("bytes"; "crypto/sha256"; "encoding/hex"; "sort"; "strconv"; "strings")

// EncodeFloatInput is the canonical JSON form of one input (input_ids and an
// all-ones attention_mask of the same length): what InputHash digests, and
// what the kit's tools print as a sample input.
func EncodeFloatInput(ids []int64) []byte {
	var b bytes.Buffer
	b.Grow(len(ids)*8 + 48)
	b.WriteString(`{"input_ids":[`)
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatInt(id, 10))
	}
	b.WriteString(`],"attention_mask":[`)
	for i := range ids {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteByte('1')
	}
	b.WriteString(`]}`)
	return b.Bytes()
}

// InputHash is the sha256 of the exact input the guest sees for ids (the
// legacy input JSON encoding, so legacy and foundation scores hash alike).
func InputHash(ids []int64) string {
	sum := sha256.Sum256(EncodeFloatInput(ids))
	return hex.EncodeToString(sum[:])
}

// InputRoot binds a set of scored inputs: the sha256 of the sorted leaves
// (each naming one input, e.g. "<child>:<row_hash>:<InputHash>"), joined by
// newlines. Order-independent, so every validator computes the same root for
// the same set; "" for no leaves.
func InputRoot(leaves []string) string {
	if len(leaves) == 0 {
		return ""
	}
	s := append([]string(nil), leaves...)
	sort.Strings(s)
	sum := sha256.Sum256([]byte(strings.Join(s, "\n")))
	return hex.EncodeToString(sum[:])
}
