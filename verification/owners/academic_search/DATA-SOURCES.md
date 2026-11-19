# Scholarly title relevance source grants

Our new code: SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0.
Original metadata and annotations retain CC0. They are not placed under SAND.

## RELISH annotation originals

Primary version: https://api.figshare.com/v2/articles/7722905/versions/1
Dataset DOI: https://doi.org/10.6084/m9.figshare.7722905.v1
Credit: Brown, Peter (2019). RELISH_v1. figshare. Dataset.
The version declares CC0 and contains one file, RELISH_v1.json.gz, file ID
14376173. Its published MD5 is 64e93f32da691bc756a25a3fa8e44451.

Full project listing page 1 contains one dataset. Page 2 is empty. This is not
a sample of a hidden evaluation pool. The source describes expert annotations
with relevant, partial and irrelevant PMID lists relative to a seed article.
These are source human annotations, not AI preference votes. We exclude partial
labels and any contradictory seed/candidate judgments. We do not read or retain
annotator identities, experience levels or account attributes in derived data.

CC0 section 2 waives rights for "any purpose whatsoever, including without
limitation commercial, advertising or promotional purposes". Section 3 supplies
the fallback licence. Section 4 does not clear third-party paper text. This
annotation file contains identifiers and relevance labels, not article passages.

## OpenAlex metadata originals

Primary API documentation: https://docs.openalex.org/how-to-use-the-api/api-overview
It states "all data is CC0". Only id, title, doi, ids and publication_year are
requested. No abstracts, linked PDFs, full paper text, author information or
machine relevance scores are requested or reused. A human RELISH label remains
distinct from an OpenAlex bibliographic fact.

Primary filtering contract: https://help.openalex.org/api/filtering/
It permits up to 100 OR values and instructs per_page=100. Every requested PMID
must match an actual returned ids.pmid. We count missing and ambiguous bindings,
reject unexpected IDs and verify the complete batch count rather than treating
the default first page as the corpus. Anonymous basic requests use the existing
free IP budget. No account, API key, paid plan or paid resource is created.

## Scope and limits

The task is finding titles of scholarly papers related to a seed paper title.
This fits ACADEMIC_SEARCH's query-to-paper purpose but is a related-title
similarity proxy rather than exhaustive search truth. Original RELISH judgments
may use information beyond titles.
Title-only ambiguity is an explicit limitation and title-visibility review must
precede any final freeze. DOI and total-result-count correctness are not gold
labels. This model cannot browse or universally verify facts.

Paper identity includes PMID, OpenAlex ID, normalized DOI and NFKC-casefolded
title. Assign complete identity components before pair construction, retain
only pairs fully inside a split and group selected seed/candidate triples for
independent-original counting. Never move one paper between splits to improve
metrics. Fit controls only on train and freeze their coefficients in bytes.
Validation may choose a candidate. Independent final audit stays unread by
model selection and is measured once by the controller.

QASPER is retained as a failed source pilot. Full query/title grouping leaves
49/6/5 independent train/validation/test components. No gate is relaxed to use
it. SciRepEval's ODC-BY collection licence alone does not clear its abstracts.
TREC-COVID has 50 topics and changing judgment history, not a new generic
query corpus. Neither has entered this candidate's data.
