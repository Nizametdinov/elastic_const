import math
import numpy as np


class Triplet(object):
    """This class describes system of 3 particles with coordinates l=[0,0], m, n"""
    def __init__(self, m, n, fem, pair_fem, fdc, pair_fdc):
        self.m = m
        self.n = n
        self.particles = {'m': m, 'n': n, 'l': [0., 0.]}
        self.positions = np.array([[0., 0.], m, n])

        self.fem = fem
        self.pair_fem = pair_fem
        self.fdc = fdc
        self.pair_fdc = pair_fdc

        r_m = math.sqrt(m[0] * m[0] + m[1] * m[1])
        r_n = math.sqrt(n[0] * n[0] + n[1] * n[1])
        r_mn = math.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)
        self.r = {'ml': r_m, 'nl': r_n, 'lm': r_m, 'ln': r_n, 'mn': r_mn, 'nm': r_mn}

        self.pair_forces = {}
        self.pair_force_derivatives = {}
        self.triplet_forces = fem.compute_forces(self.positions)
        self.triplet_force_derivatives = {}

    def f2(self, pair, coord):
        if pair not in self.pair_forces:
            self.pair_forces[pair] = self.pair_fem.compute_forces(self.r[pair])
            self.pair_forces[pair[::-1]] = self.pair_forces[pair]
        p1, p2 = pair
        return self.pair_forces[pair].rotate(to=self.particles[p1], origin=self.particles[p2])[coord - 1]

    def dF2(self, pair, coord, dparticle, dcoord):
        p1, p2 = pair
        if p1 != dparticle and p2 != dparticle:
            return 0.

        if pair not in self.pair_force_derivatives:
            self.pair_force_derivatives[pair] = self.pair_fdc.derivative_of_force(self.r[pair])
            self.pair_force_derivatives[pair[::-1]] = self.pair_force_derivatives[pair]
        if coord == 1 and dcoord == 1:
            num = 0
        elif coord == 2 and dcoord == 2:
            num = 2
        else:
            num = 1
        dF2 = self.pair_force_derivatives[pair].rotate(to=self.particles[p1], origin=self.particles[p2])[num]
        if dparticle == p1:
            return dF2
        else:
            return -dF2

    def ΔF(self, particle, coord):
        coord_letter = self._coord_letter(coord)
        p_num = self._particle_num(particle)
        pairs = self._pairs_for(particle)
        return self.triplet_forces.force(p_num, coord_letter) - self.f2(pairs[0], coord) - self.f2(pairs[1], coord)

    def dΔF(self, particle, coord, dparticle, dcoord):
        key = dparticle + str(dcoord)
        dcoord_letter = self._coord_letter(dcoord)
        dp_num = self._particle_num(dparticle)
        if key not in self.triplet_force_derivatives:
            self.triplet_force_derivatives[key] = self.fdc.derivative_of_forces(
                dcoord_letter, dp_num, self.positions
            )
        coord_letter = self._coord_letter(coord)
        p_num = self._particle_num(particle)
        pairs = self._pairs_for(particle)
        dF2_1 = self.dF2(pairs[0], coord, dparticle, dcoord)
        dF2_2 = self.dF2(pairs[1], coord, dparticle, dcoord)
        return self.triplet_force_derivatives[key].derivative(p_num, coord_letter) - dF2_1 - dF2_2

    def _coord_letter(self, coord):
        return {1: 'x', 2: 'y', 3: 'z'}[coord]

    def _particle_num(self, particle):
        return {'l': 1, 'm': 2, 'n': 3}[particle]

    def _pairs_for(self, particle):
        return {'l': ['lm', 'ln'], 'm': ['ml', 'mn'], 'n': ['nl', 'nm']}[particle]