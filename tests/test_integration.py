import unittest
import os
import sys
import numpy as np
import numpy.testing as np_test
import elastic_const.force_derivatives as fd
import elastic_const.forces as fs
from elastic_const.elastic_const_computation import compute_constants_xy_method
from unittest.mock import Mock

dirname = os.path.dirname(os.path.abspath(__file__))


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        # TODO: hide warnings
        fs.FORCE_CACHE_FILE = 'triplet_forces_3.1.txt'
        fd.DERIVATIVE_CACHE_FILE = 'computed_force_derivatives_3.1.txt'
        working_dir = os.path.join(dirname, 'test_data')
        self.pair_fem = fs.PairFemSimulation([], working_dir)
        self.triplet_fem = fs.TripletFemSimulation([], working_dir)
        self.pair_fdc = fd.PairForceDerivativeComputation(self.pair_fem, order=11)
        self.triplet_fdc = fd.ForceDerivativeComputation(working_dir, self.triplet_fem)

        self.pair_fem.cache.save_result = Mock()
        self.triplet_fem.cache.save_result = Mock()
        self.triplet_fdc.cache.save_result = Mock()

        self.stdout = sys.stdout
        sys.stdout = open(os.devnull,'w')

    def test_integration(self):
        c11, c1111, c1122, c1212 = \
            compute_constants_xy_method(3.1, 5, self.triplet_fem, self.pair_fem, self.triplet_fdc, self.pair_fdc)
        np_test.assert_almost_equal(
            np.array([c11, c1111, c1122, c1212]), np.array([-0.9214449,  4.4632044,  0.3454239,  0.6836688])
        )

        self.pair_fem.cache.save_result.assert_not_called()
        self.triplet_fem.cache.save_result.assert_not_called()
        self.triplet_fdc.cache.save_result.assert_not_called()

    def tearDown(self):
        sys.stdout = self.stdout


if __name__ == "__main__":
    unittest.main()
