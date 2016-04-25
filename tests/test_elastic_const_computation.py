import unittest
import os
import numpy as np
import elastic_const.elastic_const_computation as ecc

dirname = os.path.dirname(os.path.abspath(__file__))


class GridTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_hexagonal_lattice(self):
        expected_first_order_pairs = [
            np.array([1., 0.]), np.array([1./2, np.sqrt(3)/2]),
            np.array([-1./2, np.sqrt(3)/2]), np.array([-1., 0.]),
            np.array([-1./2, -np.sqrt(3)/2]), np.array([1./2, -np.sqrt(3)/2])
        ]  # Hexagon
        expected_second_order_pairs = [
            np.array([0., -np.sqrt(3)]), np.array([3./2, -np.sqrt(3)/2]),
            np.array([3./2, np.sqrt(3)/2]), np.array([0., np.sqrt(3)]),
            np.array([-3./2, np.sqrt(3)/2]), np.array([-3./2, -np.sqrt(3)/2])
        ]
        expected_third_order_pairs = [2 * p for p in expected_first_order_pairs]
        expected_triplets = [
            (np.array([1./2, np.sqrt(3)/2]), np.array([1., 0.])),
            (np.array([-1./2, np.sqrt(3)/2]), np.array([1./2, np.sqrt(3)/2])),
            (np.array([-1., 0.]), np.array([-1./2, np.sqrt(3)/2])),
            (np.array([-1., 0.]), np.array([-1./2, -np.sqrt(3)/2])),
            (np.array([-1./2, -np.sqrt(3)/2]), np.array([1./2, -np.sqrt(3)/2])),
            (np.array([1./2, -np.sqrt(3)/2]), np.array([1., 0.]))
        ]

        pairs, triplets = ecc.hexagonal_lattice(1, 3)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(triplets), 3)
        self.assert_contain_same_points(pairs[0], expected_first_order_pairs)
        self.assert_contain_same_points(pairs[1], expected_second_order_pairs)
        self.assert_contain_same_points(pairs[2], expected_third_order_pairs)
        self.assert_contain_same_points(triplets[0], expected_triplets)
        self.assertEqual(len(triplets[1]), 24)
        self.assertEqual(len(triplets[2]), 51)

    def test_quadratic_lattice(self):
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

        pairs, triplets = ecc.quadratic_lattice(3., 4)

        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(triplets), 4)
        self.assert_contain_same_points(pairs[0], expected_first_order_pairs)
        self.assert_contain_same_points(pairs[1], expected_second_order_pairs)
        self.assert_contain_same_points(pairs[2], expected_third_order_pairs)
        self.assertEqual(len(pairs[3]), 8)

        self.assertEqual(triplets[0], [])
        self.assert_contain_same_points(triplets[1], expected_second_order_triplets)
        self.assertEqual(len(triplets[2]), 18)
        self.assertEqual(len(triplets[3]), 72)

    def test_primitive_cubic_lattice(self):
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

        pairs, triplets = ecc.primitive_qubic_lattice(3., 4)

        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(triplets), 4)
        self.assert_contain_same_points(pairs[0], expected_first_order_pairs)
        self.assert_contain_same_points(pairs[1], expected_second_order_pairs)
        self.assertEqual(len(pairs[2]), 8)
        self.assert_contain_same_points(pairs[3], expected_fourth_order_pairs)

        self.assertEqual(triplets[0], [])
        self.assertEqual(len(triplets[1]), 60)
        self.assertEqual(len(triplets[2]), 72)
        self.assertEqual(len(triplets[3]), 81)

    def tearDown(self):
        pass

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


if __name__ == "__main__":
    unittest.main()
