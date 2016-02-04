import numpy as np

from elastic_const.misc import format_float, pairwise_distances, pairs, \
    cross_product_2d, shift_triangle, apply_transform_to_list_of_vectors
from elastic_const.renumbering import Renumbering


def _mirror_is_required(old_positions, new_positions):
    """This function checks whether pairs of vectors formed by sides of triangles have the same handedness"""
    new_sin_12 = cross_product_2d(new_positions[1] - new_positions[0], new_positions[2] - new_positions[0])
    old_sin_12 = cross_product_2d(old_positions[1] - old_positions[0], old_positions[2] - old_positions[0])
    return new_sin_12 * old_sin_12 < 0


def _rotation_matrix(from_triangle, to_triangle):
    # from_triangle[0] and to_triangle[0] must equal to [0, 0]
    particle_num = 1
    r = np.linalg.norm(to_triangle[particle_num])
    cos_theta = to_triangle[particle_num].dot(from_triangle[particle_num]) / (r * r)
    sin_theta = cross_product_2d(from_triangle[particle_num], to_triangle[particle_num]) / (r * r)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])


class TripletForces(object):
    def __init__(self, positions, forces):
        if isinstance(positions, np.ndarray):
            self.positions = np.copy(positions)
        else:
            self.positions = np.array(list(pairs(positions)))
        if isinstance(forces[0], np.ndarray):
            self.forces = np.copy(np.array(forces))
        else:
            self.forces = np.array(list(pairs(forces)))
        self.distances = sorted(pairwise_distances(self.positions))

    def force(self, particle_num, axis=None):
        """
        Returns force acting on particle with given number along given axis.
        If no axis is given returns force vector.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x', 'y' or None
        """
        if not axis:
            return self.forces[particle_num - 1]
        coord_num = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        return self.forces[particle_num - 1][coord_num]

    def __eq__(self, other):
        if not isinstance(other, TripletForces):
            return False
        return np.allclose(self.positions, other.positions) and np.allclose(self.forces, other.forces)

    def __repr__(self):
        return 'TripletForces(positions={0}, forces={1})'.format(self.positions, self.forces)

    def flat_positions(self):
        return self.positions.flatten()

    def flat_forces(self):
        return self.forces.flatten()

    def have_coords(self, positions):
        return np.allclose(self.positions, positions)

    def change_positions(self, new_positions):
        # renumber and shift
        renum = Renumbering(self.positions, new_positions)
        old_positions = renum.apply_to(self.positions)
        old_positions = shift_triangle(old_positions, -old_positions[0])
        old_forces = renum.apply_to(self.forces)

        if np.allclose(new_positions, old_positions):
            return TripletForces(new_positions, old_forces)

        assert np.allclose(new_positions[0], [0., 0.])

        # mirror
        if _mirror_is_required(old_positions, new_positions):
            mirror = np.array([[1., 0.], [0., -1.]])
            old_positions = apply_transform_to_list_of_vectors(mirror, old_positions)
            old_forces = apply_transform_to_list_of_vectors(mirror, old_forces)

        # rotate
        transform_matrix = _rotation_matrix(old_positions, new_positions)

        transformed_positions = apply_transform_to_list_of_vectors(transform_matrix, old_positions)
        assert np.allclose(new_positions, transformed_positions), \
            '{0} != {1}'.format(new_positions, transformed_positions)

        return TripletForces(new_positions, apply_transform_to_list_of_vectors(transform_matrix, old_forces))

    def to_string(self):
        return ' '.join(
            map(format_float, np.concatenate((self.flat_positions(), self.flat_forces())))
        )

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split(' ')))
        return cls(parsed[0:6], parsed[6:])
