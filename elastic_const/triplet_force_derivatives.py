from scipy.misc import derivative
from os import path
from elastic_const.cache_base import CacheBase
from elastic_const.misc import format_float, euclidean_distance, pairwise_distances, cross_product_2d, shift_triangle, \
    renumbering, reverse_renumbering, apply_renumbering, pairs
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
            return derivative_set.calculate_rotated_derivatives(axis, particle_num, positions)
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

    def derivative_of_forces(self, axis, particle_num, positions, skip_cache=False):
        """
        Returns ForceDerivatives object with f1x, f1y, f2x, f2y, f3x, f3y derivatives
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        positions: ndarray of coordinates of 3 particles np.array([[x1, y1], [x2, y2], [x3, y3]])
        """
        axis = axis.lower()
        if not skip_cache:
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

    def __init__(self, positions, triplet_fdc: TripletForceDerivativeComputation):
        self.positions = np.copy(positions)
        self.sorted_distances = sorted(pairwise_distances(positions))
        self.derivatives = {}
        self.triplet_fdc = triplet_fdc
        if self.triplet_fdc:
            self.forces = self.triplet_fdc.simulation.compute_forces(positions)

    def __particles_with_derivatives(self):
        return {i for i in range(3) if 'x' + str(i + 1) in self.derivatives}

    def __ensure_have_derivatives_for(self, particle_num):
        if 'x' + str(particle_num) not in self.derivatives:
            self.add_derivatives(self.triplet_fdc.derivative_of_forces(
                'x', particle_num, self.positions, skip_cache=True
            ))
        if 'y' + str(particle_num) not in self.derivatives:
            self.add_derivatives(self.triplet_fdc.derivative_of_forces(
                'y', particle_num, self.positions, skip_cache=True
            ))

    def __ensure_have_pair_for(self, particle_num):
        if len(self.__particles_with_derivatives()) == 1:
            self.__ensure_have_derivatives_for(particle_num % 3 + 1)  # 1 -> 2, 2 -> 3, 3 -> 1

    def add_derivatives(self, force_derivatives: TripletForceDerivatives):
        assert np.allclose(self.positions, force_derivatives.positions)
        variable = force_derivatives.axis + str(force_derivatives.particle_num)
        self.derivatives[variable] = force_derivatives

    def calculate_rotated_derivatives(self, axis, particle, positions):
        renum = renumbering(self.positions, positions)
        reverse_renum = reverse_renumbering(renum)
        new_positions = apply_renumbering(renum, positions)
        # TODO: think about renumbering for isosceles triangle

        if np.allclose(self.positions, new_positions):
            variable = axis + str(renum[particle - 1] + 1)
            derivatives = self.derivatives.get(variable, None)
            if derivatives is None:
                return None
            derivatives = apply_renumbering(reverse_renum, list(pairs(derivatives.derivatives)))
            return TripletForceDerivatives(
                axis, particle, positions, np.array(derivatives).flatten()
            )

        particle_num = particle - 1
        particle_num = renum[particle_num]

        self.__ensure_have_derivatives_for(particle_num + 1)
        self.__ensure_have_pair_for(particle_num + 1)

        pair_particle_num = (self.__particles_with_derivatives() - {particle_num}).pop()
        spare_particle = ({0, 1, 2} - {particle_num, pair_particle_num}).pop()

        coord_num = {'x': 0, 'y': 1}[axis]
        pair_coord_num = {0: 1, 1: 0}[coord_num]

        mirror = np.identity(2)  # No mirror
        new_sin_12 = cross_product_2d(new_positions[1] - new_positions[0], new_positions[2] - new_positions[0])
        old_sin_12 = cross_product_2d(self.positions[1] - self.positions[0], self.positions[2] - self.positions[0])
        if new_sin_12 * old_sin_12 < 0:
            mirror[pair_coord_num, pair_coord_num] = -1.
        new_positions = np.tensordot(new_positions, mirror, axes=1)
        old_positions = self.positions

        new_positions = shift_triangle(new_positions, -new_positions[spare_particle])
        old_positions = shift_triangle(old_positions, -old_positions[spare_particle])

        r = np.linalg.norm(new_positions[particle_num])
        cos_theta = new_positions[particle_num].dot(old_positions[particle_num]) / (r * r)
        sin_theta = cross_product_2d(old_positions[particle_num], new_positions[particle_num]) / (r * r)
        transform_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        inverse_transform = transform_matrix.T

        dcos_theta = old_positions[particle_num][coord_num] * r ** 2
        dcos_theta -= (new_positions[particle_num][coord_num] *
                       new_positions[particle_num].dot(old_positions[particle_num]))
        dcos_theta /= r ** 4
        dsin_theta = (-1) ** (coord_num + 1) * old_positions[particle_num][pair_coord_num] * r ** 2
        dsin_theta -= (new_positions[particle_num][coord_num] *
                       cross_product_2d(old_positions[particle_num], new_positions[particle_num]))
        dsin_theta /= r ** 4

        d_transform = np.array([
            [dcos_theta, -dsin_theta],
            [dsin_theta,  dcos_theta]
        ])
        d_inverse_transform = d_transform.T

        dp0 = np.array([
            d_inverse_transform.dot(new_positions[particle_num]) + inverse_transform[:, coord_num],
            d_inverse_transform.dot(new_positions[pair_particle_num])
        ])

        dfs = []
        for i in range(3):
            f0i = self.forces.force(i + 1)
            deriv_matrix = np.array([
                [self.derivatives['x' + str(particle_num + 1)].derivative(i + 1),
                 self.derivatives['y' + str(particle_num + 1)].derivative(i + 1)],
                [self.derivatives['x' + str(pair_particle_num + 1)].derivative(i + 1),
                 self.derivatives['y' + str(pair_particle_num + 1)].derivative(i + 1)]
            ])
            df0i = np.tensordot(deriv_matrix, dp0, axes=([0, 1], [0, 1]))
            dfs.append(mirror.dot(transform_matrix.dot(df0i) + d_transform.dot(f0i)))

        dfs = apply_renumbering(reverse_renum, dfs)
        return TripletForceDerivatives(
            axis, particle, positions,
            np.array([dfs[0][0], dfs[0][1], dfs[1][0], dfs[1][1], dfs[2][0], dfs[2][1]])
        )
