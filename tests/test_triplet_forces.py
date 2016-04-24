from elastic_const.triplet_forces import TripletForces
import unittest
import numpy as np
import numpy.testing as np_test


class TestTripletForces(unittest.TestCase):
    def setUp(self):
        pass

    def test_change_positions__simple_rotation(self):
        forces = TripletForces(
            np.array([[0., 0.], [0., 3.], [3., 3.]]),
            np.array([[-3.0672052, -0.3964177], [2.6707874, -2.6707874], [0.3964177, 3.0672052]])
        )
        np_test.assert_allclose(
            forces.change_positions(np.array([[0., 0.], [3., 0.], [3., -3.]])).forces,  # rotation -pi/2
            [[-0.3964177, 3.0672052], [-2.6707874, -2.6707874], [3.0672052, -0.3964177]]
        )

    def test_change_positions__mirror(self):
        forces = TripletForces(
            np.array([[0., 0.], [3., 0.], [6., 3.]]),
            np.array([[-2.7466673, -0.0163318], [2.3173516, -0.3966546], [0.4293180, 0.4129863]])
        )
        np_test.assert_allclose(
            forces.change_positions(np.array([[0., 0.], [0., 3.], [3., 6.]])).forces,
            [[-0.0163318, -2.7466673], [-0.3966546, 2.3173516], [0.4129863, 0.4293180]]
        )

    def test_change_positions__mirror_and_rotate(self):
        forces = TripletForces(
            np.array([[0., 0.], [3., 0.], [3., 3.]]),
            np.array([[-3.0672052, -0.3964177], [2.6707873, -2.6707874], [0.3964177, 3.0672052]])
        )
        np_test.assert_allclose(
            forces.change_positions(np.array([[0., 0.], [-3., 0.], [-3., 3.]])).forces,
            [[3.0672052, -0.3964177], [-2.6707874, -2.6707874], [-0.3964177, 3.0672052]]
        )

    def test_change_positions__shift_and_renumber(self):
        forces = TripletForces(
            np.array([[0., 0.], [0., 3.], [3., 3.]]),
            np.array([[-3.0672052, -0.3964177], [2.6707873, -2.6707874], [0.3964177, 3.0672052]])
        )
        np_test.assert_allclose(
            forces.change_positions(np.array([[0., 0.], [3., 0.], [0., -3.]])).forces,
            [[2.6707874, -2.6707874], [0.3964177, 3.0672052], [-3.0672052, -0.3964177]]
        )

    def test_change_positions__rotate_and_renumber(self):
        positions = np.array([[0., 0.], [3., 0.], [6., 3.]])
        force_list = np.array([[-2.7466673, -0.0163318], [2.3173516, -0.3966546], [0.4293180, 0.4129863]])
        forces = TripletForces(positions, force_list)

        transform = np.array([
            [np.cos(np.pi / 5), -np.sin(np.pi / 5)],
            [np.sin(np.pi / 5),  np.cos(np.pi / 5)]
        ])

        new_positions = np.array([[0., 0.], transform.dot(positions[2]), transform.dot(positions[1])])
        np_test.assert_allclose(
            forces.change_positions(new_positions).forces,
            [transform.dot(force_list[0]), transform.dot(force_list[2]), transform.dot(force_list[1])]
        )

    def test_change_positions__rotate_mirror_shift_and_renumber(self):
        positions = np.array([[0., 0.], [3., 0.], [6., 3.]])
        force_list = np.array([[-2.7466673, -0.0163318], [2.3173516, -0.3966546], [0.4293180, 0.4129863]])
        forces = TripletForces(positions, force_list)

        transform = np.array([
            [np.cos(2 * np.pi / 3),  np.sin(2 * np.pi / 3)],
            [np.sin(2 * np.pi / 3), -np.cos(2 * np.pi / 3)]
        ])

        new_positions = np.array([transform.dot(positions[2]), transform.dot(positions[1]), [0., 0.]])
        new_positions[2] -= new_positions[0]
        new_positions[1] -= new_positions[0]
        new_positions[0] -= new_positions[0]
        np_test.assert_allclose(
            forces.change_positions(new_positions).forces,
            [transform.dot(force_list[2]), transform.dot(force_list[1]), transform.dot(force_list[0])]
        )

    def test_change_positions__non_zero_origin(self):
        forces = TripletForces(
            np.array([[0., 0.05], [-3., 0.], [0., -3.]]),
            np.array([[2.530714, 2.672007], [-2.882504, 0.3963235], [0.3517901, -3.068331]])
        )
        np_test.assert_allclose(
            forces.change_positions(np.array([[0.05, 0.], [0., 3.], [-3., 0.]])).forces,  # rotation -pi/2
            [[2.672007, -2.530714], [0.3963235, 2.882504], [-3.068331, -0.3517901]]
        )


if __name__ == "__main__":
    unittest.main()