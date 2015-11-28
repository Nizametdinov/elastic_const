from scipy.misc import derivative
from os import path
from elastic_const.cache_base import CacheBase
from elastic_const.misc import format_float, euclidean_distance, pairwise_distances, cross_product_2d
import numpy as np
import logging

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5
DERIVATIVE_CACHE_FILE = 'computed_force_derivatives.txt'


class TripletForceDerivatives(object):
    def __init__(self, axis, particle_num, positions, derivatives):
        self.positions = np.copy(positions)
        self.particle_num = particle_num
        self.axis = axis.lower()
        self.derivatives = derivatives

    def derivative(self, particle_num, axis=None):
        """
        Returns derivative of force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x', 'y' or None
        """
        variable_num = (particle_num - 1) * 2
        if axis is None:
            return np.array(self.derivatives[variable_num:variable_num+2])
        if axis.lower() == 'y':
            variable_num += 1
        return self.derivatives[variable_num]

    def __eq__(self, other):
        if not isinstance(other, TripletForceDerivatives):
            return False
        return (self.axis == other.axis and self.particle_num == other.particle_num and
                np.allclose(self.positions, other.positions) and np.allclose(self.derivatives, other.derivatives))

    def have_coords(self, axis, particle_num, positions):
        return self.axis == axis and self.particle_num == particle_num and np.allclose(self.positions, positions)

    def __repr__(self):
        return 'ForceDerivatives("{0}", {1}, {2}, {3})'.format(
            self.axis, self.particle_num, self.positions, self.derivatives
        )

    def to_string(self):
        return '{0}{1} '.format(self.axis, self.particle_num) + ' '.join(
            map(format_float, list(self.positions.flatten()) + list(self.derivatives))
        )

    @classmethod
    def from_string(cls, string):
        variable, *numbers = string.split()
        parsed = list(map(float, numbers))
        positions = np.array(parsed[0:6])
        positions.shape = (3, 2)  # FIXME: not 3D ready
        return cls(variable[0], int(variable[1]), positions, parsed[6:])


class TripletForceDerivativeCache(CacheBase):
    """This class stores computed force derivatives"""

    def __init__(self, working_dir, triplet_fdc, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, DERIVATIVE_CACHE_FILE)
        super().__init__(cache_file_path)
        self.sets = []
        self.triplet_fdc = triplet_fdc

    def _value_from_string(self, string):
        return TripletForceDerivatives.from_string(string)

    def __find_set(self, positions):
        result = next((s for s in self.sets if np.allclose(s.positions, positions)), None)
        if result is None:
            result = TripletDerivativeSet(positions, self.triplet_fdc)
            self.sets.append(result)
        return result

    def __deduce_from_set(self, axis, particle_num, positions):
        sorted_distances = sorted(pairwise_distances(positions))
        derivative_set = next((s for s in self.sets if np.allclose(s.sorted_distances, sorted_distances)), None)
        if derivative_set:
            return derivative_set.try_deduce(axis, particle_num, positions)
        else:
            return None

    def save_result(self, value):
        super().save_result(value)
        self.__find_set(value.positions).add_derivatives(value)

    def read(self, axis, particle_num, positions):
        axis = axis.lower()
        result = next(
            (fd for fd in self.values if fd.have_coords(axis, particle_num, positions)),
            None
        )
        return result or self.__deduce_from_set(axis, particle_num, positions)


class TripletForceDerivativeComputation(object):
    def __init__(self, working_dir, simulation, order=FINITE_DIFF_ORDER, step=FINITE_DIFF_STEP, r=1.,
                 derivative_func=derivative):
        self.simulation = simulation
        self.order = order
        self.step = step
        self.cache = TripletForceDerivativeCache(working_dir, self)
        self.r = r
        self.derivative_func = derivative_func

    def derivative_of_forces(self, axis, particle_num, positions):
        """
        Returns ForceDerivatives object with f1x, f1y, f2x, f2y, f3x, f3y derivatives
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        positions: ndarray of coordinates of 3 particles np.array([[x1, y1], [x2, y2], [x3, y3]])
        """
        axis = axis.lower()
        cached = self.cache.read(axis, particle_num, positions)
        if cached:
            return cached

        axis_num = {'x': 0, 'y': 1, 'z': 2}[axis]
        var_positions = np.copy(positions)

        def force_func(arg):
            var_positions[particle_num - 1][axis_num] = arg
            result = np.array(self.simulation.compute_forces(var_positions).forces)
            logging.debug('positions = %s; forces = %s', var_positions, result)
            return result

        derivatives = self.derivative_func(
            force_func, positions[particle_num - 1][axis_num],
            dx=self.__get_step(positions, particle_num), order=self.order
        )
        result = TripletForceDerivatives(axis, particle_num, positions, derivatives)
        logging.debug(result)
        self.cache.save_result(result)
        return result

    def __get_step(self, positions, num):
        num -= 1
        p = positions[num]
        min_dist = min(euclidean_distance(p, other) for i, other in enumerate(positions) if i != num)
        if min_dist - 2 * self.r < 1.0:
            return self.step * (min_dist - 2 * self.r)
        return self.step


class TripletDerivativeSet(object):
    """This class stores set of derivatives for triplets with the save positions"""

    def __init__(self, positions, triplet_fdc):
        self.positions = np.copy(positions)
        self.sorted_distances = sorted(pairwise_distances(positions))
        self.derivatives = {}
        self.triplet_fdc = triplet_fdc
        if self.triplet_fdc:
            self.forces = self.triplet_fdc.simulation.compute_forces(positions)

    def add_derivatives(self, force_derivatives):
        assert np.allclose(self.positions, force_derivatives.positions)
        variable = force_derivatives.axis + str(force_derivatives.particle_num)
        self.derivatives[variable] = force_derivatives

    def try_deduce(self, axis, particle_num, positions):
        # Renumbering
        # positions[0] should be [0, 0]
        particle_num -= 1
        r = np.linalg.norm(positions[particle_num])
        cos_theta = positions[particle_num].dot(self.positions[particle_num]) / (r * r)
        sin_theta = cross_product_2d(self.positions[particle_num], positions[particle_num]) / (r * r)
        transform_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        inverse_transform = transform_matrix.T
        if particle_num != 0:
            coord_num = {'x': 0, 'y': 1, 'z': 2}[axis]
            pair_coord_num = {0: 1, 1: 0}[coord_num]

            dcos_theta = self.positions[particle_num][coord_num] * r ** 2
            dcos_theta -= positions[particle_num][coord_num] * positions[particle_num].dot(self.positions[particle_num])
            dcos_theta /= r ** 4
            dsin_theta = (-1) ** (coord_num + 1) * self.positions[particle_num][pair_coord_num] * r ** 2
            dsin_theta -= (positions[particle_num][coord_num] *
                           cross_product_2d(self.positions[particle_num], positions[particle_num]))
            dsin_theta /= r ** 4

            d_inverse_transform = np.array([
                [ dcos_theta, dsin_theta],
                [-dsin_theta, dcos_theta]
            ])
            d_transform = d_inverse_transform.T

            dp02 = d_inverse_transform.dot(positions[1])
            dp03 = d_inverse_transform.dot(positions[2])
            dp0 = np.array([dp02, dp03])
            dp0[particle_num - 1] += inverse_transform[:, coord_num]

            F01 = self.forces.force(1)
            F02 = self.forces.force(2)
            F03 = self.forces.force(3)

            deriv_matrix_1 = np.array([
                [self.derivatives['x2'].derivative(1), self.derivatives['y2'].derivative(1)],
                [self.derivatives['x3'].derivative(1), self.derivatives['y3'].derivative(1)]
            ])
            dF01 = np.tensordot(deriv_matrix_1, dp0, axes=([0, 1], [0, 1]))
            dF1 = transform_matrix.dot(dF01) + d_transform.dot(F01)

            deriv_matrix_2 = np.array([
                [self.derivatives['x2'].derivative(2), self.derivatives['y2'].derivative(2)],
                [self.derivatives['x3'].derivative(2), self.derivatives['y3'].derivative(2)]
            ])
            dF02 = np.tensordot(deriv_matrix_2, dp0, axes=([0, 1], [0, 1]))
            dF2 = transform_matrix.dot(dF02) + d_transform.dot(F02)

            deriv_matrix_3 = np.array([
                [self.derivatives['x2'].derivative(3), self.derivatives['y2'].derivative(3)],
                [self.derivatives['x3'].derivative(3), self.derivatives['y3'].derivative(3)]
            ])
            dF03 = np.tensordot(deriv_matrix_3, dp0, axes=([0, 1], [0, 1]))
            dF3 = transform_matrix.dot(dF03) + d_transform.dot(F03)

            return TripletForceDerivatives(
                axis, particle_num + 1, positions, np.array([dF1[0], dF1[1], dF2[0], dF2[1], dF3[0], dF3[1]])
            )

        return None
