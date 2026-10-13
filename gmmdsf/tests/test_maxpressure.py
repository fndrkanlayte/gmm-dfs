"""Small fixtures for signal mapping and upstream/downstream pressure signs."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_PKG, 'baselines')); _sys.path.insert(0, _PKG)
import unittest
from types import SimpleNamespace

from eval_maxpressure import choose_phase, phase_movements


class MaxPressureTests(unittest.TestCase):
    def test_served_movements_and_duplicate_links(self):
        phases = [SimpleNamespace(state="Ggr"), SimpleNamespace(state="rrG")]
        links = [[("a", "b", "via1"), ("a", "b", "via2")],
                 [("c", "d", "via3")], [("e", "f", "via4")]]
        self.assertEqual(phase_movements(phases, links),
                         [[("a", "b"), ("c", "d")], [("e", "f")]])

    def test_downstream_congestion_changes_selected_phase(self):
        phases = [[("a", "b")], [("c", "d")]]
        # a has more vehicles, but b is congested: select c -> d.
        self.assertEqual(choose_phase(phases, {"a": 10, "b": 12, "c": 4, "d": 0}), 1)
        self.assertEqual(choose_phase(phases, {"a": 10, "b": 0, "c": 4, "d": 0}), 0)

    def test_tie_selects_first_phase(self):
        self.assertEqual(choose_phase([[('a', 'b')], [('b', 'a')]],
                                      {'a': 0, 'b': 0}), 0)

    def test_bad_signal_mapping_fails(self):
        with self.assertRaises(ValueError):
            phase_movements([SimpleNamespace(state="GG")], [[]])
        with self.assertRaises(ValueError):
            phase_movements([SimpleNamespace(state="r")], [[("a", "b", "v")]])


if __name__ == "__main__":
    unittest.main()
