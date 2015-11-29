from elastic_const.misc import renumbering, apply_renumbering, combine_renumbering
import unittest
import numpy as np
import numpy.testing as np_test


class TestRenumbering(unittest.TestCase):
    def setUp(self):
        pass

    def test_renumbering(self):
        triangle1 = np.array([[1, 2], [3, 4], [6, 7]])
        triangle2 = np.array([[6, 7], [3, 4], [1, 2]])
        second_to_first = renumbering(triangle1, triangle2)
        np_test.assert_equal(apply_renumbering(second_to_first, triangle2), triangle1)

    def test_combine_renumbering(self):
        renumbering_1 = {0: 0, 1: 2, 2: 1}
        renumbering_2 = {0: 1, 1: 2, 2: 0}
        self.assertEqual(combine_renumbering(renumbering_1, renumbering_2), {0: 1, 1: 0, 2: 2})


if __name__ == "__main__":
    unittest.main()
