import force_derivatives as fd
import forces as fs
import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock

class TestForceDerivativeCache(unittest.TestCase):
    def setUp(self):
        f, self.cache_file = tempfile.mkstemp()
        test_data = """
            x1 0.0 0.0 1.0 1.0e-1 1.0 0.0 -1.707e-1 -1.507e-2 -1.9323 -0.53 0.1 0.2
            y1 0.0 0.0 1.0 1.0e-1 1.0 0.0 -1.032e-2 -1.112e-1 -1.232 -0.33 0.1 0.2
            y2 0.0 0.0 1.0 2.0e-1 1.0 1.0 -1.007e-1 -1.107e-2 -1.012 -0.53 0.1 0.2
        """
        with open(f, 'w') as cache_file:
            cache_file.write(test_data)
        self.subject = fd.ForceDerivativeCache(None, cache_file = self.cache_file)

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
        derivatives = fd.ForceDerivatives('x', 3, [0, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(derivatives)
        self.assertIn(derivatives, self.subject.values)

        with open(self.cache_file) as cache_file:
            lines = [line.strip() for line in cache_file if line.strip()]
            self.assertEqual(len(lines), 4)
            self.assertEqual(fd.ForceDerivatives.from_string(lines[-1]), derivatives)

    def test_read(self):
        cached = self.subject.read('y', 1, [0.0, 0.0, 1.0, 0.1, 1.0, 0.0])
        self.assertEqual(cached.axis, 'y')
        self.assertEqual(cached.particle_num, 1)
        self.assertEqual(cached.positions, [0.0, 0.0, 1.0, 0.1, 1.0, 0.0])
        self.assertEqual(cached.derivatives, [-1.032e-2, -0.1112, -1.232, -0.33, 0.1, 0.2])

        derivatives = fd.ForceDerivatives('x', 2, [0, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(derivatives)
        cached = self.subject.read('x', 2, derivatives.positions)
        self.assertEqual(cached, derivatives)

    def tearDown(self):
        os.remove(self.cache_file)


class TestForceDerivativeComputation(unittest.TestCase):
    def setUp(self):
        simulation = Mock()
        self.subject = fd.ForceDerivativeComputation(None, simulation, order = 3)

    def test_derivative_of_forces(self):
        def forces(positions):
            if positions == [0., 0., 4., 0., 2., 1.]:
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.884, 0., 7.769])
            if positions == [0., 0., 4., -0.01, 2., 1.]:
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.784, 0., 7.769])
            if positions == [0., 0., 4., 0.01, 2., 1.]:
                return fs.TripletForces(positions, [-8.347, -3.884, 8.347, -3.984, 0., 7.769])
            raise ValueError('Unexpected value of positions: {0}'.format(positions))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)

        positions = [0., 0., 4., 0., 2., 1.]
        dF_dy2 = self.subject.derivative_of_forces(2, 'Y', positions)
        expected = np.array([0., 0., 0., -10., 0., 0.])
        self.assertTrue(
            (np.round(dF_dy2, 12) == expected).all(),
            '{0} != {1}'.format(dF_dy2, expected)
        )


if __name__ == "__main__":
    unittest.main()
