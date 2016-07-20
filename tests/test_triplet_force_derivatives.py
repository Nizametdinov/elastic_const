import elastic_const.triplet_force_derivatives as fd
import elastic_const.forces as fs
import unittest
import tempfile
import os
import numpy as np
import numpy.testing as np_test
from unittest.mock import Mock, ANY


class TestTripletForceDerivativeCache(unittest.TestCase):
    def setUp(self):
        f, self.cache_file = tempfile.mkstemp()
        test_data = """
            x1 0.0 0.0 1.0 1.0e-1 1.0 0.0 -1.707e-1 -1.507e-2 -1.9323 -0.53 0.1 0.2
            y1 0.0 0.0 1.0 1.0e-1 1.0 0.0 -1.032e-2 -1.112e-1 -1.232 -0.33 0.1 0.2
            y2 0.0 0.0 1.0 2.0e-1 1.0 1.0 -1.007e-1 -1.107e-2 -1.012 -0.53 0.1 0.2
        """
        with open(f, 'w') as cache_file:
            cache_file.write(test_data)
        self.subject = fd.TripletForceDerivativeCache(None, None, cache_file=self.cache_file)

    def test_restore_cache(self):
        self.assertEqual(len(self.subject.values), 3)
        np_test.assert_equal(
            self.subject.values[0].positions,
            np.array([[0.0, 0.0], [1.0, 0.1], [1.0, 0.0]])
        )
        self.assertEqual(self.subject.values[0].particle_num, 1)
        self.assertEqual(self.subject.values[0].axis, 'x')
        np_test.assert_equal(
            self.subject.values[0].derivatives,
            [[-0.1707, -0.01507], [-1.9323, -0.53], [0.1, 0.2]]
        )
        self.assertEqual(self.subject.values[2].axis, 'y')

    def test_save_result(self):
        derivatives = fd.TripletForceDerivatives(
            'x', 3, np.array([[0, 1], [2, 3], [4, 5]]), np.array([[1.1, 2], [3, 4], [5, 6.1]])
        )
        self.subject.save_result(derivatives)
        self.assertIn(derivatives, self.subject.values)

        with open(self.cache_file) as cache_file:
            lines = [line.strip() for line in cache_file if line.strip()]
            self.assertEqual(len(lines), 4)
            self.assertEqual(fd.TripletForceDerivatives.from_string(lines[-1]), derivatives)

    def test_read(self):
        cached = self.subject.read('y', 1, np.array([[0.0, 0.0], [1.0, 0.1], [1.0, 0.0]]))
        self.assertEqual(cached.axis, 'y')
        self.assertEqual(cached.particle_num, 1)
        np_test.assert_equal(cached.positions, np.array([[0.0, 0.0], [1.0, 0.1], [1.0, 0.0]]))
        np_test.assert_equal(cached.derivatives, [[-1.032e-2, -0.1112], [-1.232, -0.33], [0.1, 0.2]])

        derivatives = fd.TripletForceDerivatives(
            'x', 2, np.array([[0, 1], [2, 3], [4, 5]]), np.array([[1.1, 2], [3, 4], [5, 6.1]])
        )
        self.subject.save_result(derivatives)
        cached = self.subject.read('x', 2, derivatives.positions)
        self.assertEqual(cached, derivatives)

    def tearDown(self):
        os.remove(self.cache_file)


class TestTripletForceDerivativeComputation(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        simulation = Mock()
        self.subject = fd.TripletForceDerivativeComputation(dirname, simulation, order=3)
        self.subject.cache.cache_file_path = os.devnull

        def forces(positions):
            if np.all(positions == np.array([[0., 0.], [4., 0.], [2., 3.]])):
                return fs.TripletForces(positions, np.array([[-8.347, -3.884], [8.347, -3.884], [0., 7.769]]))
            if np.all(positions == np.array([[0., 0.], [4., -0.01], [2., 3.]])):
                return fs.TripletForces(positions, np.array([[-8.347, -3.884], [8.347, -3.784], [0., 7.769]]))
            if np.all(positions == np.array([[0., 0.], [4., 0.01], [2., 3.]])):
                return fs.TripletForces(positions, np.array([[-8.347, -3.884], [8.347, -3.984], [0., 7.769]]))
            raise ValueError('Unexpected value of positions: {0}'.format(positions))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)

        self.positions = np.array([[0., 0.], [4., 0.], [2., 3.]])

    def test_computes_derivative_of_forces(self):
        dF_dy2 = self.subject.derivative_of_forces('Y', 2, self.positions)
        np_test.assert_allclose(dF_dy2.derivatives, [[0., 0.], [0., -10.], [0., 0.]])
        np_test.assert_equal(dF_dy2.positions, self.positions)
        self.assertEqual(dF_dy2.axis, 'y')
        self.assertEqual(dF_dy2.particle_num, 2)

    def test_computes_higher_order_derivatives(self):
        def forces(positions):
            if np.all(positions == np.array([[0., 0.], [4., -0.02], [2., 3.]])):
                return fs.TripletForces(
                    positions, np.array([[-8.347, -3.884], [8.347, -3.884 - 4e-5 - 4.8e-3], [0., 7.769]]))
            if np.all(positions == np.array([[0., 0.], [4., -0.01], [2., 3.]])):
                return fs.TripletForces(
                    positions, np.array([[-8.347, -3.884], [8.347, -3.884 - 5e-6 - 1.2e-3], [0., 7.769]]))
            if np.all(positions == np.array([[0., 0.], [4., 0.], [2., 3.]])):
                return fs.TripletForces(
                    positions, np.array([[-8.347, -3.884], [8.347, -3.884], [0., 7.769]]))
            if np.all(positions == np.array([[0., 0.], [4., 0.01], [2., 3.]])):
                return fs.TripletForces(
                    positions, np.array([[-8.347, -3.884], [8.347, -3.884 + 5e-6 - 1.2e-3], [0., 7.769]]))
            if np.all(positions == np.array([[0., 0.], [4., 0.02], [2., 3.]])):
                return fs.TripletForces(
                    positions, np.array([[-8.347, -3.884], [8.347, -3.884 + 4e-5 - 4.8e-3], [0., 7.769]]))
            raise ValueError('Unexpected value of positions: {0}'.format(positions))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)
        self.subject.order = 5

        d3F_dy2 = self.subject.derivative_of_forces('Y', 2, self.positions, n=2)
        np_test.assert_allclose(d3F_dy2.derivatives, [[0., 0.], [0., -24.], [0., 0.]], atol=1e-10)
        np_test.assert_equal(d3F_dy2.positions, self.positions)
        self.assertEqual(d3F_dy2.n, 2)
        self.assertEqual(d3F_dy2.axis, 'y')
        self.assertEqual(d3F_dy2.particle_num, 2)

        d3F_dy2 = self.subject.derivative_of_forces('Y', 2, self.positions, n=3)
        np_test.assert_allclose(d3F_dy2.derivatives, [[0., 0.], [0., 30.], [0., 0.]], atol=1e-9)
        np_test.assert_equal(d3F_dy2.positions, self.positions)
        self.assertEqual(d3F_dy2.n, 3)
        self.assertEqual(d3F_dy2.axis, 'y')
        self.assertEqual(d3F_dy2.particle_num, 2)

    def test_saves_computed_values_to_cache(self):
        computed = self.subject.derivative_of_forces('y', 2, self.positions)
        cached = self.subject.cache.read('y', 2, self.positions)
        self.assertEqual(cached, computed)

    def test_reads_cached_value(self):
        cached = fd.TripletForceDerivatives('y', 2, self.positions, np.array([[1.1, 2], [3, 4], [5, 6.1]]))
        self.subject.cache.save_result(cached)

        computed = self.subject.derivative_of_forces('y', 2, self.positions)
        self.assertEqual(cached, computed)

    def test_uses_smaller_step_for_small_dist(self):
        self.subject.derivative_func = Mock(return_value=np.array([[0, 0], [0, 0], [0, 0]]))
        self.subject.simulation.compute_forces = Mock(return_value=None)

        self.subject.derivative_of_forces('x', 2, np.array([[0, 0], [2.5, 0], [3, 3]]))
        self.subject.derivative_func.assert_called_with(ANY, 2.5, dx=(2.5 - 2) * 0.01, order=3, n=1)

    def test_computes_rotated_configuration_using_cash(self):
        p2 = np.array([3., 0.])
        p3 = np.array([2.5, 3.])
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])
        positions = np.array([[0., 0.], p2, p3])
        forces = fs.TripletForces(
            positions, np.array([[-3.20695605, -0.63937358], [3.08803874, -2.48336419], [0.11891729, 3.12273782]])
        )
        self.subject.simulation.compute_forces = Mock(return_value=forces)
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'x', 2, positions,
            np.array([[3.86246709, 1.32785211e-03], [-3.14359736, 0.660160262], [-0.718868188, -0.661486892]])
        ))
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'y', 2, positions,
            np.array([[0.05438376, -0.89292005], [0.66015376, -2.57570787], [-0.7145375, 3.46862663]])
        ))
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'x', 3, positions,
            np.array([[0.29009147, 0.59165673], [-0.71886792, -0.71453534], [0.42877687, 0.12287993]])
        ))
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'y', 3, positions,
            np.array([[0.53861846, 0.49871114], [-0.66149429, 3.46864056], [0.12287998, -3.96734753]])
        ))

        new_positions = np.array([[0., 0.], rotation_matrix.dot(p2), rotation_matrix.dot(p3)])
        computed = self.subject.derivative_of_forces('x', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[2.64949995, 2.04654145], [-3.57334059, 0.08418033], [0.92384189, -2.13072143]],
            atol=1e-5
        )


class TestTripletForceDerivatives(unittest.TestCase):
    def setUp(self):
        self.subject = fd.TripletForceDerivatives(
            'x', 2, np.array([[0, 1], [2, 3], [4, 5]]), np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        )

    def test_derivative(self):
        self.assertEqual(self.subject.derivative(1, 'x'), 0.1)
        self.assertEqual(self.subject.derivative(1, 'y'), 0.2)
        self.assertEqual(self.subject.derivative(2, 'x'), 0.3)
        self.assertEqual(self.subject.derivative(2, 'y'), 0.4)
        self.assertEqual(self.subject.derivative(3, 'x'), 0.5)
        self.assertEqual(self.subject.derivative(3, 'y'), 0.6)
        np_test.assert_equal(self.subject.derivative(1), np.array([0.1, 0.2]))
        np_test.assert_equal(self.subject.derivative(2), np.array([0.3, 0.4]))
        np_test.assert_equal(self.subject.derivative(3), np.array([0.5, 0.6]))


class TestTripletDerivativeSet(unittest.TestCase):
    def setUp(self):
        self.p2 = np.array([3., 0.])
        self.p3 = np.array([2.5, 3.])
        positions = np.array([[0., 0.], self.p2, self.p3])
        self.positions = positions
        forces = fs.TripletForces(
            positions, np.array([[-3.20695605, -0.63937358], [3.08803874, -2.48336419], [0.11891729, 3.12273782]])
        )

        triplet_fdc = Mock()
        triplet_fdc.simulation = Mock()
        triplet_fdc.simulation.compute_forces = Mock(return_value=forces)
        self.subject = fd.TripletDerivativeSet(positions, triplet_fdc)

        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'x', 2, positions,
            np.array([[3.86246709, 1.32785211e-03], [-3.14359736, 0.660160262], [-0.718868188, -0.661486892]])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'y', 2, positions,
            np.array([[0.05438376, -0.89292005], [0.66015376, -2.57570787], [-0.7145375, 3.46862663]])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'x', 3, positions,
            np.array([[0.29009147, 0.59165673], [-0.71886792, -0.71453534], [0.42877687, 0.12287993]])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'y', 3, positions,
            np.array([[0.53861846, 0.49871114], [-0.66149429, 3.46864056], [0.12287998, -3.96734753]])
        ))

    def test_the_same_coordinates(self):
        computed = self.subject.calculate_rotated_derivatives('y', 3, self.positions)
        np_test.assert_equal(
            computed.derivatives, [[0.53861846, 0.49871114], [-0.66149429, 3.46864056], [0.12287998, -3.96734753]]
        )

    def test_the_same_coordinates_no_derivative(self):
        # It should return None to prevent infinite loops
        computed = self.subject.calculate_rotated_derivatives('x', 1, self.positions)
        self.assertIsNone(computed)

    def test_rotated_y2_derivative(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('y', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[2.09958592, 0.32004116], [0.08419019, -2.14597441], [-2.18377565, 1.82592549]],
            atol=1e-5
        )

    def test_rotated_x3_derivative(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('x', 3, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[-0.14717415, 0.21875807], [0.92384142, -2.18377092], [-0.77666836, 1.96501366]],
            atol=1e-5
        )

    def test_rotated_y3_derivative(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('y', 3, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[0.16571466, 0.93597361], [-2.13073534, 1.82592165], [1.96501173, -2.76189617]],
            atol=1e-5
        )

    def test_rotated_x1_derivative(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'x', 1, self.positions,
            np.array([[-4.15255934, -0.59297549], [3.86246549, 0.054379], [0.29008511, 0.53860772]])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'y', 1, self.positions,
            np.array([[-0.59299583, 0.39421187], [0.0013343, -0.89292697], [0.59165654, 0.4987081]])
        ))

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('x', 1, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[-2.5023743, -2.26537588], [2.649552, 2.09953878], [-0.14720147, 0.16579731]],
            atol=1e-4
        )

    def test_rotated_y1_derivative(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])
        self.subject.derivatives.pop('x2')
        self.subject.derivatives.pop('y2')
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'x', 1, self.positions,
            np.array([[-4.15255934, -0.59297549], [3.86246549, 0.054379], [0.29008511, 0.53860772]])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'y', 1, self.positions,
            np.array([[-0.59299583, 0.39421187], [0.0013343, -0.89292697], [0.59165654, 0.4987081]])
        ))

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('y', 1, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[-2.26530015, -1.2560202], [2.04654162, 0.32005076], [0.21875879, 0.93596839]],
            atol=1e-4
        )

    def test_rotated_and_mirrored(self):
        transform_matrix = np.array([
            [0, 1],
            [1, 0]
        ])

        new_positions = np.array([[0., 0.], transform_matrix.dot(self.p2), transform_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('x', 3, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[0.49871114, 0.53861846], [3.46864056, -0.66149429], [-3.96734753, 0.12287998]],
            atol=1e-5
        )
        computed = self.subject.calculate_rotated_derivatives('y', 3, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[0.59165673, 0.29009147], [-0.71453534, -0.71886792], [0.12287993, 0.42877687]],
            atol=1e-5
        )

    def test_rotated_and_mirrored2(self):
        transform_matrix = np.array([
            [0.5, -np.sqrt(3) / 2],
            [np.sqrt(3) / 2, 0.5]
        ]).dot(np.array([[1, 0], [0, -1]]))

        new_positions = np.array([[0., 0.], transform_matrix.dot(self.p2), transform_matrix.dot(self.p3)])
        computed = self.subject.calculate_rotated_derivatives('x', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[0.32004116, 2.09958592], [-2.14597441, 0.08419019], [1.82592549, -2.18377565]],
            atol=1e-5
        )

        computed = self.subject.calculate_rotated_derivatives('y', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[2.04654145, 2.64949995], [0.08418033, -3.57334059], [-2.13072143, 0.92384189]],
            atol=1e-5
        )

    def test_renumbered(self):
        new_positions = np.array([[0., 0.], self.p3, self.p2])
        computed = self.subject.calculate_rotated_derivatives('x', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[0.29009147, 0.59165673], [0.42877687, 0.12287993], [-0.71886792, -0.71453534]],
            atol=1e-5
        )

    def test_rotated_and_renumbered(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])

        new_positions = np.array([rotation_matrix.dot(self.p3), [0., 0.], rotation_matrix.dot(self.p2)])
        computed = self.subject.calculate_rotated_derivatives('y', 1, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[1.96501173, -2.76189617], [0.16571466, 0.93597361], [-2.13073534, 1.82592165]],
            atol=1e-5
        )

    def test_mirrored_shifted_and_renumbered(self):
        mirror_matrix = np.array([
            [-1, 0],
            [ 0, 1]
        ])

        new_positions = np.array([[0., 0.], mirror_matrix.dot(-self.p3), mirror_matrix.dot(self.p2 - self.p3)])
        computed = self.subject.calculate_rotated_derivatives('y', 3, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [[0.7145375, 3.46862663], [-0.05438376, -0.89292005], [-0.66015376, -2.57570787]],
            atol=1e-4
        )

    def test_not_enough_data(self):
        called_with_axes = []

        def derivative_of_forces(axis, particle_num, positions, skip_cache):
            self.assertEqual(particle_num, 1)
            np_test.assert_equal(positions, self.positions)
            self.assertTrue(skip_cache)
            called_with_axes.append(axis)
            return fd.TripletForceDerivatives(
                axis, 1, self.positions,
                np.array([[-4.15255934, -0.59297549], [3.86246549, 0.054379], [0.29008511, 0.53860772]])
            )

        self.subject.triplet_fdc.derivative_of_forces = Mock(side_effect=derivative_of_forces)
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        self.subject.calculate_rotated_derivatives('x', 1, new_positions)

        self.assertEqual(called_with_axes, ['x', 'y'])

    def test_not_enough_data2(self):
        called_with_axes = []

        def derivative_of_forces(axis, particle_num, positions, skip_cache):
            self.assertEqual(particle_num, 1)
            np_test.assert_equal(positions, self.positions)
            self.assertTrue(skip_cache)
            called_with_axes.append(axis)
            return fd.TripletForceDerivatives(
                axis, 1, self.positions,
                np.array([[-4.15255934, -0.59297549], [3.86246549, 0.054379], [0.29008511, 0.53860772]])
            )

        self.subject.triplet_fdc.derivative_of_forces = Mock(side_effect=derivative_of_forces)

        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])
        self.subject.derivatives.pop('x2')
        self.subject.derivatives.pop('y2')

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        self.subject.calculate_rotated_derivatives('y', 3, new_positions)

        self.assertEqual(called_with_axes, ['x', 'y'])


if __name__ == "__main__":
    unittest.main()
