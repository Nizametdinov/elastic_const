from elastic_const.misc import format_float, order_points_by_distance, euclidean_distance, pairwise_distances, pairs
import numpy as np


def normalization_transform(triangle):
    norm_1 = np.linalg.norm(triangle[1])
    cos_a = triangle[1][0] / norm_1
    sin_a = triangle[1][1] / norm_1
    transform = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    inverse = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    if transform[1].dot(triangle[2]) < 0:
        transform[1] = -transform[1]
        inverse[:, 1] = -inverse[:, 1]
    return transform, inverse


class TripletForces(object):
    def __init__(self, positions, forces):
        if isinstance(positions[0], np.ndarray):
            self.positions = positions
        else:
            self.positions = [np.array([p1, p2]) for p1, p2 in pairs(positions)]
        if isinstance(forces[0], np.ndarray):
            self.forces = list(np.array(forces).flatten())
        else:
            self.forces = list(forces)
        self.distances = sorted(pairwise_distances(self.positions))


    def force(self, particle_num, axis):
        """
        Returns force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        """
        variable_num = (particle_num - 1) * 2
        if axis.lower() == 'y':
            variable_num += 1
        return self.forces[variable_num]

    def __eq__(self, other):
        if not isinstance(other, TripletForces):
            return False
        return np.allclose(self.positions, other.positions) and np.allclose(self.forces, other.forces)

    def __repr__(self):
        return 'TripletForces(positions={0}, forces={1})'.format(self.positions, self.forces)

    def flat_positions(self):
        return np.array(self.positions).flatten()

    def have_coords(self, positions):
        return np.allclose(self.flat_positions(), positions)

    def change_positions(self, positions):
        positions = [np.array([p1, p2]) for p1, p2 in pairs(positions)]
        nodes = list(enumerate(positions))
        ordered = order_points_by_distance(nodes, lambda p1, p2: euclidean_distance(p1[1], p2[1]))
        indices = [i for i, _ in ordered]
        shifted_points = [p - ordered[0][1] for _, p in ordered]
        _, inverse_transform = normalization_transform(shifted_points)

        denormalized_positions = [inverse_transform.dot(p) for p in self.positions]
        assert np.allclose(shifted_points, denormalized_positions), \
            '{0} != {1}'.format(shifted_points, denormalized_positions)

        denormalized_forces = [inverse_transform.dot(f) for f in pairs(self.forces)]
        forces = [None] * 3
        for i, force in zip(indices, denormalized_forces):
            forces[i] = force
        return TripletForces(positions, forces)

    def normalized(self):
        force_vecs = list(pairs(self.forces))
        points = list(zip(self.positions, force_vecs))
        # renumber
        ordered = order_points_by_distance(points, lambda p1, p2: euclidean_distance(p1[0], p2[0]))
        ordered_forces = [f for _, f in ordered]
        # use first particle as origin
        points = [p - ordered[0][0] for p, _ in ordered]
        # rotate and mirror particles
        transform, _ = normalization_transform(points)
        normalized_points = [transform.dot(p) for p in points]
        normalized_forces = [transform.dot(f) for f in ordered_forces]
        return TripletForces(normalized_points, normalized_forces)

    def to_string(self):
        return ' '.join(
            map(format_float, np.concatenate((self.flat_positions(), self.forces)))
        )

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split(' ')))
        return cls(parsed[0:6], parsed[6:])