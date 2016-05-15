import unittest
import os
import numpy as np
import elastic_const.crystals as cs

dirname = os.path.dirname(os.path.abspath(__file__))


class GridTest(unittest.TestCase):
    def assert_contain_same_points(self, lst1, lst2):
        def find_missing(lst, in_lst):
            missing = []
            for p1 in lst:
                if next((p2 for p2 in in_lst if np.allclose(p1, p2)), None) is None:
                    missing.append(p1)
            return missing

        missing = find_missing(lst1, lst2)
        extra = find_missing(lst2, lst1)
        self.assertTrue(
            len(missing) == 0 and len(extra) == 0,
            "Expected lists \n  {0}\n and \n  {1}\n to have the same elements. "
            "\nMissing elements: {2}\nExtra elements: {3}".format(lst1, lst2, missing, extra)
        )


class QuadraticLatticeTest(GridTest):
    def setUp(self):
        self.subject = cs.QuadraticLattice(3., 5)

    def test_ws_cell_volume(self):
        self.assertEqual(self.subject.ws_cell_volume(), 3. * 3.)

    def test_points_for(self):
        expected_first_order_pairs = [
            np.array([3., 0.]), np.array([0., 3.]),
            np.array([-3., 0.]), np.array([0., -3.])
        ]
        expected_second_order_pairs = [
            np.array([3., 3.]), np.array([-3., 3.]),
            np.array([-3., -3.]), np.array([3., -3.])
        ]
        expected_third_order_pairs = [
            np.array([6., 0.]), np.array([0., 6.]),
            np.array([-6., 0.]), np.array([0., -6.])
        ]

        self.assert_contain_same_points(self.subject.points_for(0), expected_first_order_pairs)
        self.assert_contain_same_points(self.subject.points_for(1), expected_second_order_pairs)
        self.assert_contain_same_points(self.subject.points_for(2), expected_third_order_pairs)
        self.assertEqual(len(self.subject.points_for(3)), 8)

    def test_pairs_for(self):
        expected_second_order_triplets = [
            (3 * np.array([0., 1.]), 3 * np.array([1., 0.])),
            (3 * np.array([0., -1.]), 3 * np.array([1., 0.])),
            (3 * np.array([-1., 0.]), 3 * np.array([0., -1.])),
            (3 * np.array([-1., 0.]), 3 * np.array([0., 1.])),
            (3 * np.array([1., 1.]), 3 * np.array([1., 0.])),
            (3 * np.array([1., 1.]), 3 * np.array([0., 1.])),
            (3 * np.array([-1., 1.]), 3 * np.array([0., 1.])),
            (3 * np.array([-1., 1.]), 3 * np.array([-1., 0.])),
            (3 * np.array([-1., -1.]), 3 * np.array([-1., 0.])),
            (3 * np.array([-1., -1.]), 3 * np.array([0., -1.])),
            (3 * np.array([1., -1.]), 3 * np.array([0., -1.])),
            (3 * np.array([1., -1.]), 3 * np.array([1., 0.])),
        ]
        self.assertEqual(self.subject.pairs_for(0), [])
        self.assert_contain_same_points(self.subject.pairs_for(1), expected_second_order_triplets)
        self.assertEqual(len(self.subject.pairs_for(2)), 18)
        self.assertEqual(len(self.subject.pairs_for(3)), 72)


class HexagonalLatticeTest(GridTest):
    def setUp(self):
        self.subject = cs.HexagonalLattice(3., 5)

    def test_ws_cell_volume(self):
        self.assertEqual(self.subject.ws_cell_volume(), 3. * 3. * np.sqrt(3) / 2)

    def test_points_for(self):
        expected_first_order_pairs = [
            3 * np.array([1., 0.]), 3 * np.array([1. / 2, np.sqrt(3) / 2]),
            3 * np.array([-1. / 2, np.sqrt(3) / 2]), 3 * np.array([-1., 0.]),
            3 * np.array([-1. / 2, -np.sqrt(3) / 2]), 3 * np.array([1. / 2, -np.sqrt(3) / 2])
        ]  # Hexagon
        expected_second_order_pairs = [
            3 * np.array([0., -np.sqrt(3)]), 3 * np.array([3. / 2, -np.sqrt(3) / 2]),
            3 * np.array([3. / 2, np.sqrt(3) / 2]), 3 * np.array([0., np.sqrt(3)]),
            3 * np.array([-3. / 2, np.sqrt(3) / 2]), 3 * np.array([-3. / 2, -np.sqrt(3) / 2])
        ]
        expected_third_order_pairs = [2 * p for p in expected_first_order_pairs]

        self.assert_contain_same_points(self.subject.points_for(0), expected_first_order_pairs)
        self.assert_contain_same_points(self.subject.points_for(1), expected_second_order_pairs)
        self.assert_contain_same_points(self.subject.points_for(2), expected_third_order_pairs)
        self.assertEqual(len(self.subject.points_for(3)), 12)

    def test_pairs_for(self):
        expected_triplets = [
            (3 * np.array([1. / 2, np.sqrt(3) / 2]), 3 * np.array([1., 0.])),
            (3 * np.array([-1. / 2, np.sqrt(3) / 2]), 3 * np.array([1. / 2, np.sqrt(3) / 2])),
            (3 * np.array([-1., 0.]), 3 * np.array([-1. / 2, np.sqrt(3) / 2])),
            (3 * np.array([-1., 0.]), 3 * np.array([-1. / 2, -np.sqrt(3) / 2])),
            (3 * np.array([-1. / 2, -np.sqrt(3) / 2]), 3 * np.array([1. / 2, -np.sqrt(3) / 2])),
            (3 * np.array([1. / 2, -np.sqrt(3) / 2]), 3 * np.array([1., 0.]))
        ]
        self.assert_contain_same_points(self.subject.pairs_for(0), expected_triplets)
        self.assertEqual(len(self.subject.pairs_for(1)), 24)
        self.assertEqual(len(self.subject.pairs_for(2)), 51)


class PrimitiveCubicLatticeTest(GridTest):
    def setUp(self):
        self.subject = cs.PrimitiveCubicLattice(3., 5)

    def test_ws_cell_volume(self):
        self.assertEqual(self.subject.ws_cell_volume(), 3. * 3. * 3.)

    def test_points_for(self):
        expected_first_order_pairs = [
            np.array([3., 0., 0.]),
            np.array([-3., 0., 0.]),
            np.array([0., 3., 0.]),
            np.array([0., -3., 0.]),
            np.array([0., 0., 3.]),
            np.array([0., 0., -3.])
        ]
        expected_second_order_pairs = [
            np.array([3., 3., 0.]),
            np.array([3., -3., 0.]),
            np.array([-3., 3., 0.]),
            np.array([-3., -3., 0.]),
            np.array([3., 0., 3.]),
            np.array([3., 0., -3.]),
            np.array([-3., 0., 3.]),
            np.array([-3., 0., -3.]),
            np.array([0., 3., 3.]),
            np.array([0., 3., -3.]),
            np.array([0., -3., 3.]),
            np.array([0., -3., -3.])
        ]
        expected_fourth_order_pairs = [
            np.array([6., 0., 0.]),
            np.array([-6., 0., 0.]),
            np.array([0., 6., 0.]),
            np.array([0., -6., 0.]),
            np.array([0., 0., 6.]),
            np.array([0., 0., -6.])
        ]

        self.assert_contain_same_points(self.subject.points_for(0), expected_first_order_pairs)
        self.assert_contain_same_points(self.subject.points_for(1), expected_second_order_pairs)
        self.assertEqual(len(self.subject.points_for(2)), 8)
        self.assert_contain_same_points(self.subject.points_for(3), expected_fourth_order_pairs)

    def test_pairs_for(self):
        self.assertEqual(self.subject.pairs_for(0), [])
        self.assertEqual(len(self.subject.pairs_for(1)), 60)
        self.assertEqual(len(self.subject.pairs_for(2)), 72)
        self.assertEqual(len(self.subject.pairs_for(3)), 81)

if __name__ == "__main__":
    unittest.main()
