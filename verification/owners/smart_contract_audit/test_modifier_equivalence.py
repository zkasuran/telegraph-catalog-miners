# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import unittest

from fresh_sources import training


class ModifierChecks(unittest.TestCase):
    def test_parameterless_caller_guard_precedes_literal_writer(self):
        for index in (0, 2, 4, 5):
            for vulnerable in (True, False):
                entity = 340000 + index * 10
                source, facts = training(index, entity, vulnerable)
                self.assertIn(f"modifier only{entity}(){{", source)
                self.assertIn(f"external only{entity}{{", source)
                self.assertLess(source.index("_;}"), source.index(f"function f{entity}"))
                self.assertEqual(facts["signature"], f"f{entity}(uint256)")
                self.assertEqual(facts["argument"], 17)
