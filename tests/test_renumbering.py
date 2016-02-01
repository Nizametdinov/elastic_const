from elastic_const.renumbering import Renumbering, _combine_renumbering
import unittest
import numpy as np
import numpy.testing as np_test


class TestRenumbering(unittest.TestCase):
    def setUp(self):
        pass

    def test_renumbering(self):
        triangle1 = np.array([[1, 2], [5, 5], [5, 7]])
        triangle2 = np.array([[5, 5], [5, 7], [1, 2]])
        first_to_second = Renumbering(triangle1, triangle2)
        np_test.assert_equal(first_to_second.apply_to(triangle1), triangle2)
        np_test.assert_equal(first_to_second.apply_reverse_to(triangle2), triangle1)

    def test_renumbering2(self):
        triangle1 = np.array([[1, 2], [3, 4], [5, 7]])
        triangle2 = np.array([[3, 4], [5, 7], [1, 2]])
        first_to_second = Renumbering(triangle1, triangle2)
        print(first_to_second._renumbering)
        np_test.assert_equal(first_to_second.apply_to(triangle1), triangle2)
        np_test.assert_equal(first_to_second.apply_reverse_to(triangle2), triangle1)

    def test_renumbering3(self):
        triangle1 = np.array([[0, 0], [4, 3], [4, 0]])
        triangle2 = np.array([[4, 3], [4, 0], [0, 0]])
        first_to_second = Renumbering(triangle1, triangle2)
        np_test.assert_equal(first_to_second.apply_to(triangle1), triangle2)
        np_test.assert_equal(first_to_second.apply_reverse_to(triangle2), triangle1)

    def test_apply_to_index(self):
        triangle1 = np.array([[0, 0], [4, 3], [4, 0]])
        triangle2 = np.array([[4, 4], [4, 1], [0, 1]])
        first_to_second = Renumbering(triangle1, triangle2)
        self.assertEqual(first_to_second.apply_to_index(0), 2)
        self.assertEqual(first_to_second.apply_to_index(1), 0)
        self.assertEqual(first_to_second.apply_to_index(2), 1)

    def test_combine_renumbering(self):
        renumbering_1 = {0: 0, 1: 2, 2: 1}
        renumbering_2 = {0: 1, 1: 2, 2: 0}
        self.assertEqual(_combine_renumbering(renumbering_1, renumbering_2), {0: 1, 1: 0, 2: 2})


if __name__ == "__main__":
    unittest.main()
