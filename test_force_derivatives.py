import force_derivatives as fd
import forces as fs
import unittest
import tempfile
import os
from unittest.mock import Mock

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
        dFy2_dy2 = self.subject.derivative_of_force(2, 'Y', positions)
        self.assertEqual(round(dFy2_dy2, 12), -10.)


if __name__ == "__main__":
    unittest.main()
