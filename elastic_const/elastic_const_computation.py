import math
import numpy as np
from elastic_const.misc import euclidean_distance


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


class Configuration(object):
    """This class describes system of 3 particles with coordinates l=[0,0], m, n"""
    def __init__(self, m, n, fem, pair_fem, fdc, pair_fdc):
        self.m = m
        self.n = n
        self.particles = {'m': m, 'n': n, 'l': [0., 0.]}

        self.fem = fem
        self.pair_fem = pair_fem
        self.fdc = fdc
        self.pair_fdc = pair_fdc

        r_m = math.sqrt(m[0] * m[0] + m[1] * m[1])
        r_n = math.sqrt(n[0] * n[0] + n[1] * n[1])
        r_mn = math.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)
        self.r = {'ml': r_m, 'nl': r_n, 'mn': r_mn, 'nm': r_mn}

        self.pair_forces = {}
        self.pair_force_derivatives = {}
        self.triplet_forces = fem.compute_forces([0, 0, m[0], m[1], n[0], n[1]])
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
                dcoord_letter, dp_num, [0, 0, self.m[0], self.m[1], self.n[0], self.n[1]]
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
        return {'m': ['ml', 'mn'], 'n': ['nl', 'nm']}[particle]


def c_αβστ_m(pair_conf, α, β, σ, τ):
    dF2_m_σ_dm_τ = pair_conf.dF2(σ, τ)
    c = pair_conf.m[α - 1] * pair_conf.m[β - 1] * dF2_m_σ_dm_τ
    if α == β:
        force2_m_β = pair_conf.f2(β)
        c -= pair_conf.m[α - 1] * force2_m_β
    return c


def c_αβστ_mn(conf, α, β, σ, τ):
    dΔF_m_β_dm_τ = conf.dΔF('m', β, 'm', τ)
    dΔF_n_β_dm_τ = conf.dΔF('n', β, 'm', τ)

    dΔF_m_β_dn_τ = conf.dΔF('m', β, 'n', τ)
    dΔF_n_β_dn_τ = conf.dΔF('n', β, 'n', τ)
    # check if dΔF_n_β_dm_τ == dΔF_m_β_dn_τ when β == τ

    c  = conf.m[α - 1] * conf.m[σ - 1] * dΔF_m_β_dm_τ
    c += conf.m[α - 1] * conf.n[σ - 1] * dΔF_m_β_dn_τ
    c += conf.n[α - 1] * conf.m[σ - 1] * dΔF_n_β_dm_τ
    c += conf.n[α - 1] * conf.n[σ - 1] * dΔF_n_β_dn_τ

    if β == τ:
        ΔF_m_α = conf.ΔF('m', α)
        ΔF_n_α = conf.ΔF('n', α)
        c -= conf.m[σ - 1] * ΔF_m_α + conf.n[σ - 1] * ΔF_n_α

    return c


def compute_constants(fem, pair_fem, fdc, pair_fdc):
    a = 3.0
    v0 = a * a

    first_order = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    second_order = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    c11 = 0
    c1111 = 0
    c1122 = 0
    c1212 = 0

    for i, m in enumerate(first_order):
        m = a * np.array(m)
        pair_conf = Pair(m, np.array([0, 0]), pair_fem, pair_fdc)
        force2_m_x = pair_conf.f2(1)

        c11 += m[0] * force2_m_x
        c1111 += c_αβστ_m(pair_conf, 1, 1, 1, 1)
        c1122 += c_αβστ_m(pair_conf, 1, 1, 2, 2)
        c1212 += c_αβστ_m(pair_conf, 1, 2, 1, 2)

        for j, n in enumerate(first_order):
            if j <= i:
                continue
            n = a * np.array(n)
            print('m =', m, '; n =', n)

            conf = Configuration(m, n, fem, pair_fem, fdc, pair_fdc)

            ΔF_m_x = conf.ΔF('m', 1)
            ΔF_n_x = conf.ΔF('n', 1)

            c11 += m[0] * ΔF_m_x + m[1] * ΔF_n_x

            c1111 += c_αβστ_mn(conf, 1, 1, 1, 1)
            c1122 += c_αβστ_mn(conf, 1, 1, 2, 2)
            c1212 += c_αβστ_mn(conf, 1, 2, 1, 2)

    return c11 / v0, c1111 / v0, c1122 / v0, c1212 / v0