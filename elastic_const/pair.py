from elastic_const.misc import euclidean_distance

__author__ = 'azatnizametdinov'


class Pair(object):
    def __init__(self, p1, p2, pair_fem, pair_fdc):
        self.p1 = p1
        self.p2 = p2
        self.m = p1
        self.pair_fem = pair_fem
        self.pair_fdc = pair_fdc
        self.r = euclidean_distance(p1, p2)
        self.pair_force = None
        self.pair_force_derivative = None

    def f2(self, coord):
        if not self.pair_force:
            self.pair_force = self.pair_fem.compute_forces(self.r)
        return self.pair_force.rotate(to=self.p1, origin=self.p2)[coord - 1]

    def dF2(self, coord, dcoord):
        if not self.pair_force_derivative:
            self.pair_force_derivative = self.pair_fdc.derivative_of_force(self.r)
        if coord == 1 and dcoord == 1:
            num = 0
        elif coord == 2 and dcoord == 2:
            num = 2
        else:
            num = 1
        return self.pair_force_derivative.rotate(to=self.p1, origin=self.p2)[num]