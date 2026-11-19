# Threat model

| Threat | Defence | Test or gate |
|---|---|---|
| Unlicensed original text | Per-article URI, authors, DOI and retained notice | Original source smoke plus remote re-fetch |
| Tampered source bytes | Git blob and SHA-256 checks | Source tamper test |
| XML entity expansion | 4 MiB ceiling, explicit entity declarations refused | Entity and overflow controls |
| Split leakage | DOI/page/text components before selection | Normalized overlap failure controls |
| Answer-only style or length shortcut | Mirrored questions, frozen train-only counts and unchanged primary probe | Training/validation source gates |
| Invisible evidence | Complete prompt plus answers, no token cropping | Visibility gate and evidence removal controls |
| Normalized stand-in wire | Literal render, raw answer matching, JSON/CRLF controls | Actual CLI and wire tests |
| Wrong class or runtime identity | Fresh primary class snapshot and full parser | Go mutation tests and remote identity controls |
| Local model operation | Remote guard before foundation or neural imports | Local entrypoint refusal test |
| Failed checkpoint licence loss | Inert safetensors and ZIP stamps plus attribution | Stamp roundtrip tests, remote Torch roundtrip |
| Slow or oversized source/runtime | Socket timeout, file/count caps, fixed runner timeout | Overflow tests and bounded workflow |

## Known limits

Source agreement is not scientific correctness. The symbolic grammar is narrow
and refuses unrelated wording, citations without evidence and unsupported claim
relations. The neural candidate is prepared but not remotely tested. A Go build
and the actual executor must pass before any package is accepted.

The caller that chooses a live judgment response body is a separate trust
boundary. A raw plain-string answer and a JSON wrapper are different strings.
No source-held-out model score certifies their live equivalence. Citation counts
and hedging do not establish truth. This corpus has one publisher, 112 original
groups and constructed labels. It does not measure publisher transfer or broad
research-answer quality. The controller alone owns publication and qualification.

## Release boundaries

The standalone verify.sh exports OWNER_SOURCE_ONLY and supplies sitecustomize.py
to its subprocesses. Neural imports are rejected before module loading. A guard
installed by the action adapter remains first on PYTHONPATH. The source gate does
not weaken the remote runner identity checks or permit local model operations.

The full checkpoint artifact can exceed the registration adapter's 16 MiB
evidence ceiling. A successful model/runtime job therefore uploads the validated
registration-ready.json alone as a separate artifact. The failed checkpoint
archive remains preserved with original grants but is never represented as a
passing registration report. No source-held-out report certifies global truth.

## Released asset gate

The release gate hashes the exact published delta and original registration-ready report, checks their actual run/commit binding and verifies retained in-byte terms and attribution. It reads only bounded static metadata and opaque binary bytes, never a local neural model. The live action adapter still checks fresh source classes, public reference vectors, token refusal and exact package identity before registering.
