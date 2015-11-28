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
        self.assertEqual(
            self.subject.values[0].derivatives,
            [-0.1707, -0.01507, -1.9323, -0.53, 0.1, 0.2]
        )
        self.assertEqual(self.subject.values[2].axis, 'y')

    def test_save_result(self):
        derivatives = fd.TripletForceDerivatives('x', 3, np.array([[0, 1], [2, 3], [4, 5]]), [1.1, 2, 3, 4, 5, 6.1])
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
        self.assertEqual(cached.derivatives, [-1.032e-2, -0.1112, -1.232, -0.33, 0.1, 0.2])

        derivatives = fd.TripletForceDerivatives('x', 2, np.array([[0, 1], [2, 3], [4, 5]]), [1.1, 2, 3, 4, 5, 6.1])
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
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.884, 0., 7.769])
            if np.all(positions == np.array([[0., 0.], [4., -0.01], [2., 3.]])):
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.784, 0., 7.769])
            if np.all(positions == np.array([[0., 0.], [4., 0.01], [2., 3.]])):
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.984, 0., 7.769])
            raise ValueError('Unexpected value of positions: {0}'.format(positions))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)

        self.positions = np.array([[0., 0.], [4., 0.], [2., 3.]])

    def test_computes_derivative_of_forces(self):
        dF_dy2 = self.subject.derivative_of_forces('Y', 2, self.positions)
        expected = np.array([0., 0., 0., -10., 0., 0.])
        self.assertTrue(
            (np.round(dF_dy2.derivatives, 12) == expected).all(),
            '{0} != {1}'.format(dF_dy2, expected)
        )
        np_test.assert_equal(dF_dy2.positions, self.positions)
        self.assertEqual(dF_dy2.axis, 'y')
        self.assertEqual(dF_dy2.particle_num, 2)

    def test_saves_computed_values_to_cache(self):
        computed = self.subject.derivative_of_forces('y', 2, self.positions)
        cached = self.subject.cache.read('y', 2, self.positions)
        self.assertEqual(cached, computed)

    def test_reads_cached_value(self):
        cached = fd.TripletForceDerivatives('y', 2, self.positions, np.array([1.1, 2, 3, 4, 5, 6.1]))
        self.subject.cache.save_result(cached)

        computed = self.subject.derivative_of_forces('y', 2, self.positions)
        self.assertEqual(cached, computed)

    def test_uses_smaller_step_for_small_dist(self):
        self.subject.derivative_func = Mock(return_value=np.array([0, 0, 0, 0, 0, 0]))
        self.subject.simulation.compute_forces = Mock(return_value=None)

        self.subject.derivative_of_forces('x', 2, np.array([[0, 0], [2.5, 0], [3, 3]]))
        self.subject.derivative_func.assert_called_with(ANY, 2.5, dx=(2.5 - 2) * 0.01, order=3)

    def test_computes_rotated_configuration_using_cash(self):
        p2 = np.array([3., 0.])
        p3 = np.array([2.4, 3.])
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])
        positions = np.array([[0., 0.], p2, p3])
        forces = fs.TripletForces(
            positions, np.array([-3.20695605, -0.63937358, 3.08803874, -2.48336419, 0.11891729, 3.12273782])
        )
        self.subject.simulation.compute_forces = Mock(return_value=forces)
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'x', 2, positions,
            np.array([3.86246709, 1.32785211e-03, -3.14359736, 0.660160262, -0.718868188, -0.661486892])
        ))
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'y', 2, positions,
            np.array([0.05438376, -0.89292005, 0.66015376, -2.57570787, -0.7145375, 3.46862663])
        ))
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'x', 3, positions,
            np.array([0.29009147, 0.59165673, -0.71886792, -0.71453534, 0.42877687, 0.12287993])
        ))
        self.subject.cache.save_result(fd.TripletForceDerivatives(
            'y', 3, positions,
            np.array([0.53861846, 0.49871114, -0.66149429, 3.46864056, 0.12287998, -3.96734753])
        ))

        new_positions = np.array([[0., 0.], rotation_matrix.dot(p2), rotation_matrix.dot(p3)])
        computed = self.subject.derivative_of_forces('x', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [2.64949995, 2.04654145, -3.57334059, 0.08418033, 0.92384189, -2.13072143],
            atol=0.1
        )


class TestTripletForceDerivatives(unittest.TestCase):
    def setUp(self):
        self.subject = fd.TripletForceDerivatives(
            'x', 2, np.array([[0, 1], [2, 3], [4, 5]]), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
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
        self.p3 = np.array([2.4, 3.])
        positions = np.array([[0., 0.], self.p2, self.p3])
        forces = fs.TripletForces(
            positions, np.array([-3.20695605, -0.63937358, 3.08803874, -2.48336419, 0.11891729, 3.12273782])
        )

        triplet_fdc = Mock()
        triplet_fdc.simulation = Mock()
        triplet_fdc.simulation.compute_forces = Mock(return_value=forces)
        self.subject = fd.TripletDerivativeSet(positions, triplet_fdc)

        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'x', 2, positions,
            np.array([3.86246709, 1.32785211e-03, -3.14359736, 0.660160262, -0.718868188, -0.661486892])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'y', 2, positions,
            np.array([0.05438376, -0.89292005, 0.66015376, -2.57570787, -0.7145375, 3.46862663])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'x', 3, positions,
            np.array([0.29009147, 0.59165673, -0.71886792, -0.71453534, 0.42877687, 0.12287993])
        ))
        self.subject.add_derivatives(fd.TripletForceDerivatives(
            'y', 3, positions,
            np.array([0.53861846, 0.49871114, -0.66149429, 3.46864056, 0.12287998, -3.96734753])
        ))

    def test_rotated_y2_derivative(self):
        rotation_matrix = np.array([
            [np.sqrt(3) / 2, -0.5],
            [0.5, np.sqrt(3) / 2]
        ])

        new_positions = np.array([[0., 0.], rotation_matrix.dot(self.p2), rotation_matrix.dot(self.p3)])
        computed = self.subject.try_deduce('y', 2, new_positions)
        np_test.assert_allclose(
            computed.derivatives, [2.09958592, 0.32004116, 0.08419019, -2.14597441, -2.18377565, 1.82592549],
            atol=0.1
        )


if __name__ == "__main__":
    unittest.main()
