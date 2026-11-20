# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic proposal-search controls. Fixtures are never scientific preferences."""
import unittest

from expand_proposals import expand, literal_presence
from proposal_matching import matching
from source import contains


def fixture(identity, annotations, text="Photosynthesis and respiration.", group=None, topics=None):
    return {"doi": identity, "text": text, "literal_keywords": annotations,
            "group_id": group or identity, "topics": topics or ["Plant Biology"]}


class ExpansionTests(unittest.TestCase):
    def test_distinct_annotation_witnesses_without_shared_keyword(self):
        rows = [fixture("synthetic-A", ["Photosynthesis"]), fixture("synthetic-B", ["respiration"])]
        proposals, report = expand(rows, [])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["status"], "unreviewed")
        self.assertEqual(proposals[0]["annotation_witness_a"]["doi"], "synthetic-A")
        self.assertEqual(report["source_pairs_examined"], 1)
        preserved, _ = expand(rows, proposals)
        self.assertEqual(preserved, proposals)

    def test_source_group_topic_and_morphological_controls(self):
        base = fixture("synthetic-A", ["Photosynthesis"])
        for other in (fixture("synthetic-B", ["respiration"], group="synthetic-A"),
                      fixture("synthetic-B", ["respiration"], topics=["Neuroscience"]),
                      fixture("synthetic-B", ["respiration"], text="Photosynthetic respiration.")):
            self.assertEqual(expand([base, other], [])[0], [])

    def test_nested_unicode_matches_equal_primary_literal_rule(self):
        vocabulary = {"dna", "dna damage", "dna damage response", "β-catenin", "navigation", "dna damage responses"}
        passage = "DNA damage response; β-catenin. Navigational bias."
        self.assertEqual(literal_presence(passage, vocabulary), {p for p in vocabulary if contains(passage, p)})

    def test_short_original_scientific_symbols_remain_unreviewed(self):
        rows = [fixture("synthetic-A", ["DNA"], text="DNA and RNA."), fixture("synthetic-B", ["RNA"], text="DNA and RNA.")]
        self.assertEqual(expand(rows, [])[0], [])
        proposals, _ = expand(rows, [], True)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["status"], "unreviewed")

    def test_packing_never_approves_or_reuses_answer_resources(self):
        rows = [fixture(name, [], group=name) for name in "ABCD"]
        for row in rows:
            row["component_topics"] = ["Neuroscience"]
        proposals = [{"proposal_id": "synthetic-1", "doi_a": "A", "doi_b": "B", "term_a": "alpha", "term_b": "beta"},
                     {"proposal_id": "synthetic-2", "doi_a": "C", "doi_b": "D", "term_a": "alpha", "term_b": "gamma"}]
        reviews = {"schema_version": 1, "reviews": [], "approved_contrasts": 0, "approved_pairs": 0}
        report = matching(rows, proposals, reviews, "test")
        self.assertEqual(report["packed_unreviewed_contrasts"], 1)
        self.assertEqual(report["selected"][0]["status"], "unreviewed")
        self.assertEqual(report["by_prospective_split"], {"test": 1})
        decision = {"proposal_id": "synthetic-1", "status": "approve", "review_origin": "AI-assisted", "choice_a": "term_a",
                    "choice_b": "term_b", "reason_a": "Synthetic A", "reason_b": "Synthetic B"}
        prior = {**reviews, "reviews": [decision], "approved_contrasts": 1, "approved_pairs": 2}
        self.assertEqual(matching(rows, proposals, prior)["packed_unreviewed_contrasts"], 0)


if __name__ == "__main__":
    unittest.main()
