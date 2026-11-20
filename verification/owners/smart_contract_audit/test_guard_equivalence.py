# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Training-only manual guards implement the same checked arithmetic domain."""
import unittest


class GuardChecks(unittest.TestCase):
    def test_composite_guards_equal_native_checked_domain_on_complete_small_width_grid(self):
        # Complete small-width state spaces, not model inference or a fitted probe.
        for width in (3, 4, 5):
            m = (1 << width) - 1
            for x in range(m + 1):
                for y in range(m + 1):
                    manual = (
                        x <= m-y and x+y <= m//3,
                        (x == 0 or x <= m//x) and x*x <= m-y,
                        y < m and x <= m//(y+1),
                        x < m and y < m and x+1 <= m//(y+1),
                        x <= m//2 and x+x <= m-y,
                        (y == 0 or x <= m//y) and x*y <= m-x,
                    )
                    native = (
                        x+y <= m and (x+y)*3 <= m,
                        x*x <= m and x*x+y <= m,
                        y+1 <= m and x*(y+1) <= m,
                        x+1 <= m and y+1 <= m and (x+1)*(y+1) <= m,
                        x+x <= m and x+x+y <= m,
                        x*y <= m and x*y+x <= m,
                    )
                    self.assertEqual(manual, native, (width, x, y))
