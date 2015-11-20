import elastic_const.triplet_force_derivatives as fd
import elastic_const.forces as fs
import unittest
import tempfile
import os
import numpy as np
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
        self.subject = fd.TripletForceDerivativeCache(None, cache_file=self.cache_file)

    def test_restore_cache(self):
        self.assertEqual(len(self.subject.values), 3)
        self.assertEqual(
            self.subject.values[0].positions,
            [0.0, 0.0, 1.0, 0.1, 1.0, 0.0]
        )
        self.assertEqual(self.subject.values[0].particle_num, 1)
        self.assertEqual(self.subject.values[0].axis, 'x')
        self.assertEqual(
            self.subject.values[0].derivatives,
            [-0.1707, -0.01507, -1.9323, -0.53, 0.1, 0.2]
        )
        self.assertEqual(self.subject.values[2].axis, 'y')

    def test_save_result(self):
        derivatives = fd.TripletForceDerivatives('x', 3, [0, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(derivatives)
        self.assertIn(derivatives, self.subject.values)

        with open(self.cache_file) as cache_file:
            lines = [line.strip() for line in cache_file if line.strip()]
            self.assertEqual(len(lines), 4)
            self.assertEqual(fd.TripletForceDerivatives.from_string(lines[-1]), derivatives)

    def test_read(self):
        cached = self.subject.read('y', 1, [0.0, 0.0, 1.0, 0.1, 1.0, 0.0])
        self.assertEqual(cached.axis, 'y')
        self.assertEqual(cached.particle_num, 1)
        self.assertEqual(cached.positions, [0.0, 0.0, 1.0, 0.1, 1.0, 0.0])
        self.assertEqual(cached.derivatives, [-1.032e-2, -0.1112, -1.232, -0.33, 0.1, 0.2])

        derivatives = fd.TripletForceDerivatives('x', 2, [0, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
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
            if positions == [0., 0., 4., 0., 2., 3.]:
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.884, 0., 7.769])
            if positions == [0., 0., 4., -0.01, 2., 3.]:
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.784, 0., 7.769])
            if positions == [0., 0., 4., 0.01, 2., 3.]:
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.984, 0., 7.769])
            raise ValueError('Unexpected value of positions: {0}'.format(positions))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)

        self.positions = [0., 0., 4., 0., 2., 3.]

    def test_computes_derivative_of_forces(self):
        dF_dy2 = self.subject.derivative_of_forces('Y', 2, self.positions)
        expected = np.array([0., 0., 0., -10., 0., 0.])
        self.assertTrue(
            (np.round(dF_dy2.derivatives, 12) == expected).all(),
            '{0} != {1}'.format(dF_dy2, expected)
        )
        self.assertEqual(dF_dy2.positions, self.positions)
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

        self.subject.derivative_of_forces('x', 2, [0, 0, 2.5, 0, 3, 3])
        self.subject.derivative_func.assert_called_with(ANY, 2.5, dx=(2.5 - 2) * 0.01, order=3)


class TestTripletForceDerivatives(unittest.TestCase):
    def setUp(self):
        self.subject = fd.TripletForceDerivatives('x', 2, [0, 1, 2, 3, 4, 5], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

    def test_derivative(self):
        self.assertEqual(self.subject.derivative(1, 'x'), 0.1)
        self.assertEqual(self.subject.derivative(1, 'y'), 0.2)
        self.assertEqual(self.subject.derivative(2, 'x'), 0.3)
        self.assertEqual(self.subject.derivative(2, 'y'), 0.4)
        self.assertEqual(self.subject.derivative(3, 'x'), 0.5)
        self.assertEqual(self.subject.derivative(3, 'y'), 0.6)

if __name__ == "__main__":
    unittest.main()
