# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import unittest

from contracts import FAMILY
from class_binding import validate_classes


class LiveClassTests(unittest.TestCase):
    def values(self):
        return ({"resolve": {"intent": FAMILY, "class": "judgment", "scorer": "judgment", "routed": True}},
                {"intent_id": FAMILY + "_GENERAL", "class": "judgment", "evaluation": {"kind": "judgment"}})

    def test_matching_primary_family_child_classes(self):
        validate_classes(*self.values())

    def test_class_and_route_are_not_guessed_or_normalized(self):
        family, child = self.values()
        for field, value in (("class", "hybrid"), ("intent", "OTHER"), ("routed", False), ("scorer", "generic")):
            bad = copy.deepcopy(family)
            bad["resolve"][field] = value
            with self.assertRaises(ValueError):
                validate_classes(bad, child)
        with self.assertRaises(ValueError):
            validate_classes(family, {**child, "class": "hybrid"})


if __name__ == "__main__":
    unittest.main()
