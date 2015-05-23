from derivatives_with_distance import *
import unittest
import numpy as np

class TestPotential3DistanceDerivatives(unittest.TestCase):
    def setUp(self):
        pass

    def test_renumber(self):
        derivatives = Potential3DistanceDerivatives(1, 2, 3, 1, 2, 3)
        renumbered = derivatives.renumber(2, 3, 1)
        self.assertEqual(renumbered.r12, 2)
        self.assertEqual(renumbered.r13, 3)
        self.assertEqual(renumbered.r23, 1)
        self.assertEqual(renumbered.derivatives(), (2, 3, 1))


if __name__ == "__main__":
    unittest.main()
