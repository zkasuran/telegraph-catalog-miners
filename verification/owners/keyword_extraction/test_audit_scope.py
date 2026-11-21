# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual audit caller statements with synthetic scores only, never neural work."""
import ast
import math
from pathlib import Path
import struct
import sys
import unittest

from audit_remote import FORMATS, metrics


def bit(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def caller_smoke(source):
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    assignments = [node for node in ast.walk(main) if isinstance(node, ast.Assign)]
    shortcuts = next(node for node in assignments if ast.unparse(node.targets[0]) == 'shortcuts')
    measured = next(node for node in assignments if ast.unparse(node.targets[0]) == 'measured[label]')
    # Compile the actual two caller statements together, preserving Python's
    # comprehension symbol resolution. Calling metrics directly missed this bug.
    smoke = ast.parse('def smoke(format_shortcuts, row_views, streams, label):\n pass\n').body[0]
    smoke.body = [shortcuts, measured, ast.parse('return shortcuts, measured[label]').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[smoke], type_ignores=[]))
    namespace = {'metrics': metrics, 'FORMATS': FORMATS, 'measured': {}}
    exec(compile(module, 'actual-audit-caller-statements', 'exec'), namespace)
    rows = [{'chosen': 'synthetic-a', 'rejected': 'synthetic-b', 'group_id': 'synthetic-group',
             'label_kind': 'synthetic-preference'}]
    return namespace['smoke']({name: {'synthetic-shortcut': .5} for name in FORMATS},
                              {name: rows for name in FORMATS},
                              [{name: [bit(.75), bit(.25)] for name in FORMATS}], 'candidate')


class AuditScopeTests(unittest.TestCase):
    def test_actual_main_metric_path(self):
        shortcuts, measured = caller_smoke(Path(__file__).with_name('audit_remote.py').read_text())
        self.assertEqual(set(measured), set(FORMATS))
        self.assertEqual(shortcuts, {name + ':synthetic-shortcut': .5 for name in FORMATS})
        for report in measured.values():
            self.assertEqual(report['pairwise_accuracy'], 1)
            self.assertEqual(report['labels']['synthetic-preference']['n'], 1)
            self.assertEqual(report['strata']['tie']['accuracy'], 1)

    @unittest.skipIf(sys.version_info < (3, 12), 'Python 3.12 public runner comprehension regression')
    def test_original_shadow_mutation_fails(self):
        source = Path(__file__).with_name('audit_remote.py').read_text()
        self.assertIn('for name, format_metric in format_shortcuts.items()', source)
        mutated = source.replace('for name, format_metric in format_shortcuts.items()',
                                 'for name, metrics in format_shortcuts.items()').replace(
                                     'for key, value in format_metric.items()', 'for key, value in metrics.items()')
        with self.assertRaises(UnboundLocalError):
            caller_smoke(mutated)

    def test_raw_bits_retained_before_aggregation(self):
        main = next(n for n in ast.parse(Path(__file__).with_name('audit_remote.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        nodes = list(ast.walk(main))
        save = next(n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'write' and 'audit-score-bits.json' in ast.unparse(n))
        aggregate = next(n for n in nodes if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'measured[label]')
        repeat = next(n for n in nodes if isinstance(n, ast.Raise) and 'fresh-process FP32 bits differ' in ast.unparse(n))
        self.assertLess(repeat.lineno, save.lineno)
        self.assertLess(save.lineno, aggregate.lineno)

    def test_score_failures_and_ties_remain_losses(self):
        rows = [{'chosen': 'a', 'rejected': 'b', 'group_id': 'synthetic', 'label_kind': 'synthetic'}]
        self.assertEqual(metrics(rows, [bit(.5), bit(.5)])['correct'], 0)
        for values in ([], [bit(.5)], [bit(math.inf), bit(.5)], [bit(math.nan), bit(.5)]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                metrics(rows, values)


if __name__ == '__main__':
    unittest.main()
