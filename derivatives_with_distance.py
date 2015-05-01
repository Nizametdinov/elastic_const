from misc import format_float, equal_vectors
import math

class Potential2DistanceDerivatives(object):
    def __init__(self, order, r, derivative):
        self.order = order
        self.r = r
        self.derivative = derivative

    def to_string():
        return str(order) + ' ' + format_float(r) + format_float(derivative)

class Potential3DistanceDerivatives(object):
    def __init__(self, order, pair_nums, r12, r13, r23, derivative):
        self.pair_nums = list(pair_nums)
        self.r12 = r12
        self.r13 = r13
        self.r23 = r23
        self.derivative = derivative
        self.distances = np.array(sorted([r12, r13, r23]))

    def have_distances(r1, r2, r3):
        return equal_vectors(self.distances, np.array(sorted([r1, r2, r3])))

    def to_string(self):
        return ' '.join(map(str, self.pair_nums)) + ' '.join(
            map(format_float, [self.r12, self.r13, self.r23. self.derivative])
        )

    def renumber(self, r12, r13, r23):
        pass

    @classmethod
    def from_string(cls, string):
        particle_num, *numbers = string.split()
        parsed = list(map(float, numbers))
        return cls(int(particle_num), *parsed)

def potential2_derivative(self, x, y):
    distance = x * x + y * y
    # compute force
    force = 0
    return force / (2 * sqrt(distance))

def potential2_second_derivative(self, df2_dr2, x, y):
    distance = x * x + y * y
    force_derivative = 0
    return (force_derivative - df2_dr2) / (4 * distance)
