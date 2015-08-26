from elastic_const.misc import format_float, EPSILON, pairwise_distances, wage_product
from elastic_const.forces import PairFemSimulation, TripletFemSimulation
from elastic_const.force_derivatives import ForceDerivativeComputation, PairForceDerivativeComputation
from elastic_const.elastic_const_computation import Configuration
from collections import namedtuple
from os import path
import elastic_const.cache_base as cache_base
import numpy as np
import math

POTENTIAL_3_DERIVATIVE_CACHE_FILE = 'f3_derivatives_cache.txt'


class Potential2DistanceDerivative(object):
    def __init__(self, order, r, derivative):
        self.order = order
        self.r = r
        self.derivative = derivative

    def to_string(self):
        return str(self.order) + ' ' + format_float(self.r) + format_float(self.derivative)


class Potential3DistanceDerivatives(
        namedtuple('F3DistanceDerivative', ['r12', 'r13', 'r23', 'df3_dr12', 'df3_dr13', 'df3_dr23'])):
    def distances(self):
        if not hasattr(self, '__distances'):
            self.__distances = np.array(sorted([self.r12, self.r13, self.r23]))
        return self.__distances

    def derivatives(self):
        if not hasattr(self, '__derivatives'):
            self.__derivatives = (self.df3_dr12, self.df3_dr13, self.df3_dr23)
        return self.__derivatives

    def have_distances(self, r1, r2, r3):
        return np.allclose(self.distances(), np.array(sorted([r1, r2, r3])))

    def to_string(self):
        return ' '.join(
            map(format_float, [self.r12, self.r13, self.r23.self.df3_dr12, self.df3_dr13, self.df3_dr23])
        )

    def renumber(self, r12, r13, r23):
        old_rs = {self.r12: 12, self.r13: 13, self.r23: 23}
        new_to_old = {12: old_rs[r12], 13: old_rs[r13], 23: old_rs[r23]}
        derivatives = {12: self.df3_dr12, 13: self.df3_dr13, 23: self.df3_dr23}
        df3_dr12, df3_dr13, df3_dr23 = (
            derivatives[new_to_old[12]], derivatives[new_to_old[13]], derivatives[new_to_old[23]]
        )
        return Potential3DistanceDerivatives(r12, r13, r23, df3_dr12, df3_dr13, df3_dr23)

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split()))
        return cls(*parsed)


class Potential3DistanceDerivativeCache(cache_base.CacheBase):
    """
    This class stores computed derivatives of three-body potintial with respect to the squared distance
    """

    def __init__(self, working_dir, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, POTENTIAL_3_DERIVATIVE_CACHE_FILE)
        super().__init__(cache_file_path)

    def _value_from_string(self, string):
        return Potential3DistanceDerivatives.from_string(string)

    def read(self, distances):
        return next((f for f in self.values if f.have_distances(*distances)), None)


class PotentialDerivativesComputation(object):
    def __init__(self, pair_fem_command, triplet_fem_command, working_dir):
        self.pair_fem = PairFemSimulation(pair_fem_command, working_dir)
        self.pair_derivative_computation = PairForceDerivativeComputation(self.pair_fem)

        self.triplet_fem = TripletFemSimulation(triplet_fem_command, working_dir)
        self.triplet_derivative_computation = ForceDerivativeComputation(working_dir, self.triplet_fem)

        self.f3_derivative_cache = Potential3DistanceDerivativeCache('')

    def potential2_derivative(self, x, y):
        distance = math.sqrt(x * x + y * y)
        force = self.pair_fem.compute_forces(distance)
        derivative = - force / (2 * distance)
        result = Potential2DistanceDerivative(1, distance, derivative)
        return result

    def potential2_second_derivative(self, first_derivative, x, y):
        distance2 = x * x + y * y
        distance = math.sqrt(distance2)
        force_derivative = self.pair_derivative_computation.derivative_of_force(distance)
        derivative = - (force_derivative + 2 * first_derivative) / (4 * distance2)
        result = Potential2DistanceDerivative(2, distance, derivative)
        return result

    def potential3_derivative(self, positions):
        p1, p2, p3 = positions
        distance12, distance23, distance13 = pairwise_distances(positions)

        cached = self.f3_derivative_cache.read(distance12, distance13, distance23)
        if cached:
            return cached.renumber(distance12, distance13, distance23)

        sorted_dist = sorted([distance12, distance23, distance13])
        if np.allclose(sorted_dist[0] + sorted_dist[1], sorted_dist[2]):
            return self.potential3_derivative_aligned(distance12, distance23, distance13)

        force2_12 = self.pair_fem.compute_forces(distance12).rotate(p2, origin=p1)
        force2_13 = self.pair_fem.compute_forces(distance13).rotate(p3, origin=p1)

        force2_23_0 = self.pair_fem.compute_forces(distance23)
        force2_32 = force2_23_0.rotate(p2, origin=p3)

        forces3 = self.triplet_fem.compute_forces(positions) # FIXME: wrong argument
        delta_force3_2 = forces3.force(2) - force2_12 - force2_32

        delta = 2 * wage_product(p2 - p1, p2 - p3)
        assert abs(delta) > EPSILON

        df3_dr12_2 = wage_product(p2 - p3, delta_force3_2) / delta
        df3_dr23_2 = wage_product(delta_force3_2, p2 - p1) / delta

        force2_23 = force2_23_0.rotate(p3, origin=p2)

        forces3 = self.triplet_fem.compute_forces(positions)
        delta_force3_3 = forces3.force(3) - force2_13 - force2_23

        delta = 2 * wage_product(p3 - p1, p3 - p2)
        assert abs(delta) > EPSILON

        df3_dr13_2 = wage_product(p3 - p2, delta_force3_3) / delta

        result = Potential3DistanceDerivatives(distance12, distance13, distance23,
                                               df3_dr12_2, df3_dr13_2, df3_dr23_2)
        return result

    def potential3_derivative_aligned(self, r12, r23, r13):
        x = 1
        x1, x2, x3 = 0., r12, r13
        m = np.array([x2, 0.])
        n = np.array([x3, 0.])
        conf = Configuration(m, n, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                             self.pair_derivative_computation)
        df3_dr23_2 = - conf.ΔF('m', x) / (4 * (x2 - x3))
        df3_dr12_2 = - conf.ΔF('m', x) / (4 * (x2 - x1))
        df3_dr13_2 = - conf.ΔF('n', x) / (4 * (x2 - x1))

        result = Potential3DistanceDerivatives(r12, r13, r23, df3_dr12_2, df3_dr13_2, df3_dr23_2)
        return result

    def potential3_second_derivative(self, positions, first_derivatives):
        x1, y1, x2, y2, x3, y3 = positions
        delta = 64 * (x2 - x1) ** 3 * (y2 - y3) ** 3 - 64 * (x2 - x3) ** 3 * (y2 - y1) ** 3
        delta += 192 * (x2 - x1) * (x2 - x3) ** 2 * (y2 - y1) ** 2 * (y2 - y3)
        delta -= 192 * (x2 - x1) ** 2 * (x2 - x3) * (y2 - y1) * (y2 - y3) ** 2

        distance_12 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        distance_13 = (x3 - x1) ** 2 + (y3 - y1) ** 2
        distance_23 = (x3 - x2) ** 2 + (y3 - y2) ** 2

        dfull_force_dx2 = self.triplet_derivative_computation.derivative_of_forces('x', 2, positions)
        dfull_force_dy2 = self.triplet_derivative_computation.derivative_of_forces('y', 2, positions)

        dforce2_12_dr = self.pair_derivative_computation.derivative_of_force(distance_12)
        dforce2_12_dx2, dforce2_12_dy2 = dforce2_12_dr.rotate(x2 - x1, y2 - y1)
        dforce2_23_dr = self.pair_derivative_computation.derivative_of_force(distance_23)
        dforce2_23_dx2, dforce2_23_dy2 = dforce2_12_dr.rotate(x2 - x3, y2 - y3)
        b1 = dfull_force_dx2.derivatives[2] - dforce2_12_dx2 - dforce2_23_dx2
        b1 -= 2 * first_derivatives.df3_dr12 + 2 * first_derivatives.df3_dr23

        # TODO: check force2 derivatives
        b2 = dfull_force_dy2.derivatives[2] - dforce2_12_dy2 - dforce2_23_dy2

        b3 = dfull_force_dy2.derivatives[3] - dforce2_12_dy2 - dforce2_23_dy2
        b3 -= 2 * first_derivatives.df3_dr12 + 2 * first_derivatives.df3_dr23
