# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic parser controls. No synthetic sentence is a translation example."""
import bz2
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from source_probe import components, link_join, norm, records, sentences, sha


class SourceProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def archive(self, data, name="links.csv"):
        path = self.root / "links.tar.bz2"
        with tarfile.open(path, "w:bz2") as archive:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        return path

    def test_lf_is_only_record_separator(self):
        self.assertEqual(next(records(io.BytesIO("1\teng\ta\u0085b\tdate\n".encode()), 4))[1][2], "a\u0085b")

    def test_oversized_record_refused(self):
        with self.assertRaises(ValueError):
            list(records(io.BytesIO(b"a" * 9000), 4))

    def test_source_hash_mismatch_refused(self):
        path = self.root / "eng.bz2"
        path.write_bytes(bz2.compress(b"1\teng\toriginal\tdate\n"))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            sentences(path, "eng")

    def test_original_text_kept_and_duplicate_identity_refused(self):
        path = self.root / "eng.bz2"
        path.write_bytes(bz2.compress("1\teng\t Original e\u0301 \tdate\n".encode()))
        with patch.dict("source_probe.SENTENCES", {"eng": sha(path)}):
            self.assertEqual(sentences(path, "eng")[1]["text"], " Original e\u0301 ")
        path.write_bytes(bz2.compress(b"1\teng\ta\td\n1\teng\tb\td\n"))
        with patch.dict("source_probe.SENTENCES", {"eng": sha(path)}), self.assertRaises(ValueError):
            sentences(path, "eng")

    def test_reciprocal_identity_measured_not_assumed(self):
        originals = {1: {"language": "eng"}, 2: {"language": "fra"}}
        report, _, pairs, _ = link_join(self.archive(b"1\t2\n2\t1\n"), originals)
        self.assertEqual(pairs, [(1, 2)])
        self.assertEqual(report["missing_reciprocals"], [])
        report, _, _, _ = link_join(self.archive(b"1\t2\n"), originals)
        self.assertEqual(report["missing_reciprocals"], [(1, 2)])

    def test_duplicate_edge_and_wrong_member_refused(self):
        originals = {1: {"language": "eng"}, 2: {"language": "fra"}}
        with self.assertRaises(ValueError):
            link_join(self.archive(b"1\t2\n1\t2\n"), originals)
        with self.assertRaises(ValueError):
            link_join(self.archive(b"1\t2\n", "../escape"), originals)

    def test_global_paths_and_normalized_aliases_merge(self):
        originals = {1: {"language": "eng", "text": "alpha"},
                     2: {"language": "fra", "text": "beta"},
                     3: {"language": "eng", "text": " ＡＬＰＨＡ "},
                     4: {"language": "fra", "text": "gamma"}}
        # ID 8 is outside the selected sentence exports. Its path still merges.
        path = self.archive(b"1\t2\n2\t1\n3\t4\n4\t3\n2\t8\n8\t4\n")
        result = components(path, originals, [(1, 2), (3, 4)], id_ceiling=10)
        self.assertEqual(result["independent_link_components"], 1)
        self.assertEqual(result["groups"][0]["cc0_original_ids"], [1, 2, 3, 4])
        self.assertEqual(norm(" ＡＬＰＨＡ "), "alpha")

    def test_graph_ceiling_refused(self):
        with self.assertRaisesRegex(ValueError, "ID ceiling"):
            components(self.archive(b"1\t11\n"), {}, [], id_ceiling=10)


if __name__ == "__main__":
    unittest.main()
