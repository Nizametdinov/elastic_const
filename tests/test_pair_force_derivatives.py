import elastic_const.pair_force_derivatives as fd
import elastic_const.forces as fs
import unittest
import numpy as np
import numpy.testing as np_test
from unittest.mock import Mock, ANY


class TestPairForceDerivativeComputation(unittest.TestCase):
    def setUp(self):
        simulation = Mock()
        self.subject = fd.PairForceDerivativeComputation(simulation, order=3, r=1.)

    def test_computes_derivative_of_force(self):
        def forces(distance):
            if distance == 3.:
                return fs.PairForce(distance, 0.865)
            if distance == 2.99:
                return fs.PairForce(distance, 0.875)
            if distance == 3.01:
                return fs.PairForce(distance, 0.855)
            raise ValueError('Unexpected value of distance: {0}'.format(distance))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)

        dF_dr = self.subject.derivative_of_force(3.)
        np_test.assert_allclose(dF_dr.derivative, -1.)
        self.assertEqual(dF_dr.n, 1)

    def test_computes_higher_order_derivative(self):
        def forces(distance):
            if np.allclose(distance, 3.08):
                return fs.PairForce(distance, 2.435255176201)
            if np.allclose(distance, 3.09):
                return fs.PairForce(distance, 2.401507749016)
            if np.allclose(distance, 3.10):
                return fs.PairForce(distance, 2.368339710530)
            if np.allclose(distance, 3.11):
                return fs.PairForce(distance, 2.335737663240)
            if np.allclose(distance, 3.12):
                return fs.PairForce(distance, 2.303688779845)
            raise ValueError('Unexpected value of distance: {0}'.format(distance))

        mock_config = {'compute_forces.side_effect': forces}
        self.subject.simulation.configure_mock(**mock_config)
        self.subject.order = 5

        dF_dr = self.subject.derivative_of_force(3.1, n=2)
        np_test.assert_allclose(dF_dr.derivative, 5.659436791668)
        self.assertEqual(dF_dr.n, 2)

        dF_dr = self.subject.derivative_of_force(3.1, n=3)
        np_test.assert_allclose(dF_dr.derivative, -13.112402000415)
        self.assertEqual(dF_dr.n, 3)

    def test_uses_smaller_step_for_small_dist(self):
        mock_config = {'compute_forces.return_value': fs.PairForce(0., 0.)}
        self.subject.simulation.configure_mock(**mock_config)
        self.subject.derivative_func = Mock(return_value=0)

        self.subject.derivative_of_force(2.5)
        self.subject.derivative_func.assert_called_with(ANY, 2.5, dx=(2.5-2)*0.01, n=1, order=3)


if __name__ == "__main__":
    unittest.main()
