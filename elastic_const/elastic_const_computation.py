import logging
import traceback
import math

import numpy as np

from itertools import product
from elastic_const.misc import pairwise_distances
from elastic_const.pair import Pair
from elastic_const.triplet import Triplet


def compute_constants(fem, pair_fem, fdc, pair_fdc):
    try:
        a = 3.0
        v0 = a * a

        first_order = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        second_order = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        rest = [
            [2, 0], [2, 1], [2, 2], [1, 2], [0, 2],
            [-1, 2], [-2, 2], [-2, 1], [-2, 0],
            [-2, -1], [-2, -2], [-1, -2], [0, -2],
            [1, -2], [2, -2], [2, -1]
        ]
        particles = a * np.array(first_order + second_order + rest)
        c11, c1111, c1122, c1212 = _pair_constants(particles, v0, pair_fdc, pair_fem)

        pairs = [[m, n] for i, m in enumerate(particles[:-1]) for n in particles[i + 1:]]
        tri_c11, tri_c1111, tri_c1122, tri_c1212 = _three_body_constants(pairs, v0, fem, pair_fem, fdc, pair_fdc)

        return c11 + tri_c11, c1111 + tri_c1111, c1122 + tri_c1122, c1212 + tri_c1212
    except:
        logging.critical('elastic_const.compute_constants error %s', traceback.format_exc())
        raise


def compute_constants_v2(fem, pair_fem, fdc, pair_fdc, a):
    """
    Incremental computation of elastic constants for 2d crystal with simple quadratic lattice
    """
    orders = [1, math.sqrt(2), 2, math.sqrt(5), 2 * math.sqrt(2)]#, 3]
    tol = 1.0e-4

    def select_pairs(a, order):
        max_dist = a * orders[order] + tol
        x = [a * i for i in range(-round(orders[order]), round(orders[order]) + 1)]
        return [np.array(p) for p in product(x, x) if np.linalg.norm(p) <= max_dist and not np.allclose(p, [0, 0])]

    def correct_triangle(p1, p2, max_dist):
        dists = np.array(pairwise_distances([np.array([0, 0]), np.array(p1), np.array(p2)]))
        return np.all(dists <= max_dist) and np.all(dists > tol)

    def select_triplets(a, order):
        max_dist = a * orders[order]
        x = [a * i for i in range(-round(orders[order]), round(orders[order]) + 1)]
        return [[np.array(p1), np.array(p2)]
                for p1 in product(x, x) for p2 in product(x, x) if correct_triangle(p1, p2, max_dist) and p1 < p2]

    try:
        v0 = a * a
        c11, c1111, c1122, c1212 = 0, 0, 0, 0
        tri_c11, tri_c1111, tri_c1122, tri_c1212 = 0, 0, 0, 0
        for order in range(len(orders)):
            print('order =', order + 1)
            particles = select_pairs(a, order)
            c11, c1111, c1122, c1212 = _pair_constants(particles, v0, pair_fdc, pair_fem)
            c11, c1111, c1122, c1212 = c11 / 2, c1111 / 2, c1122 / 2, c1212 / 2

            pairs = select_triplets(a, order)
            tri_c11, tri_c1111, tri_c1122, tri_c1212 = _three_body_constants(pairs, v0, fem, pair_fem, fdc, pair_fdc)
            tri_c11, tri_c1111, tri_c1122, tri_c1212 = tri_c11 / 3, tri_c1111 / 3, tri_c1122 / 3, tri_c1212 / 3

            logging.info(
                'Pairs order={0} C11={1}, C1111={2}, C1122={3}, C1212={4}'.format(order + 1, c11, c1111, c1122, c1212)
            )
            logging.info(
                'Triplets order={0} C11={1}, C1111={2}, C1122={3}, C1212={4}'.format(
                    order + 1, tri_c11, tri_c1111, tri_c1122, tri_c1212)
            )

        return c11 + tri_c11, c1111 + tri_c1111, c1122 + tri_c1122, c1212 + tri_c1212
    except:
        logging.critical('elastic_const.compute_constants error %s', traceback.format_exc())
        raise


def c_αβ_m(pair_conf, α, β):
    force2_m_β = pair_conf.f2(β)
    c = - pair_conf.m[α - 1] * force2_m_β
    logging.debug('m = {m}; F2_m_{β} = {f2}; ΔC{α}{β} = {c}'.format(m=pair_conf.m, α=α, β=β, f2=force2_m_β, c=c))
    return c


def c_αβστ_m(pair_conf, α, β, σ, τ):
    dF2_m_σ_dm_τ = pair_conf.dF2(σ, τ)
    c = pair_conf.m[α - 1] * pair_conf.m[β - 1] * dF2_m_σ_dm_τ
    logging.debug('m = {m}; dF2_m_{σ}_dm_{τ} = {dF}; ΔC{α}{β}{σ}{τ} = {c}'
                  .format(m=pair_conf.m, α=α, β=β, σ=σ, τ=τ, dF=dF2_m_σ_dm_τ, c=-c))
    if α == β:
        force2_m_β = pair_conf.f2(β)
        c -= pair_conf.m[α - 1] * force2_m_β
        logging.debug('m = {m}; F2_m_{β} = {F}; ΔC{α}{β}{σ}{τ} = {c}'
                      .format(m=pair_conf.m, α=α, β=β, σ=σ, τ=τ, F=force2_m_β, c=-c))
    return -c


def c_αβ_mn(conf, α, β):
    ΔF_m_β = conf.ΔF('m', β)
    ΔF_n_β = conf.ΔF('n', β)
    c = - conf.m[α - 1] * ΔF_m_β - conf.n[α - 1] * ΔF_n_β
    logging.debug(
        'm = {m}, n = {n}; ΔF_m_{β} = {Fm}; ΔF_n_{β} = {Fn}; ΔC{α}{β} = {c}'.format(
            m=conf.m, n=conf.n, α=α, β=β, Fm=ΔF_m_β, Fn=ΔF_n_β, c=c
        )
    )
    return c


def c_αβστ_mn(conf, α, β, σ, τ):
    dΔF_m_β_dm_τ = conf.dΔF('m', β, 'm', τ)
    dΔF_n_β_dm_τ = conf.dΔF('n', β, 'm', τ)

    dΔF_m_β_dn_τ = conf.dΔF('m', β, 'n', τ)
    dΔF_n_β_dn_τ = conf.dΔF('n', β, 'n', τ)

    if β == τ and not np.allclose(dΔF_n_β_dm_τ, dΔF_m_β_dn_τ):
        logging.warning('dΔF_m_{β}_dn_{τ} = {dF1} != dΔF_n_{β}_dm_{τ} = {dF2}'.format(
            β=β, τ=τ, dF1=dΔF_m_β_dn_τ, dF2=dΔF_n_β_dm_τ))

    c = conf.m[α - 1] * conf.m[σ - 1] * dΔF_m_β_dm_τ
    c += conf.m[α - 1] * conf.n[σ - 1] * dΔF_m_β_dn_τ
    c += conf.n[α - 1] * conf.m[σ - 1] * dΔF_n_β_dm_τ
    c += conf.n[α - 1] * conf.n[σ - 1] * dΔF_n_β_dn_τ

    logging.debug('m = {}, n = {}'.format(conf.m, conf.n))
    logging.debug(
        'dΔF_m_{β}_dm_{τ} = {dF1}; dΔF_m_{β}_dn_{τ} = {dF2}'.format(β=β, τ=τ, dF1=dΔF_m_β_dm_τ, dF2=dΔF_m_β_dn_τ))
    logging.debug(
        'dΔF_n_{β}_dn_{τ} = {dF1}; dΔF_n_{β}_dm_{τ} = {dF2}'.format(β=β, τ=τ, dF1=dΔF_n_β_dn_τ, dF2=dΔF_n_β_dm_τ))
    logging.debug('m = {m}, n = {n}; ΔC{α}{β}{σ}{τ} = {c}'.format(m=conf.m, n=conf.n, α=α, β=β, σ=σ, τ=τ, c=-c))
    if β == τ:
        ΔF_m_α = conf.ΔF('m', α)
        ΔF_n_α = conf.ΔF('n', α)
        c -= conf.m[σ - 1] * ΔF_m_α + conf.n[σ - 1] * ΔF_n_α
        logging.debug('ΔF_m_{α} = {Fm}; ΔF_n_{α} = {Fn}'.format(α=α, Fm=ΔF_m_α, Fn=ΔF_n_α))
        logging.debug('m = {m}, n = {n}; ΔF ΔC{α}{β}{σ}{τ} = {c}'.format(m=conf.m, n=conf.n, α=α, β=β, σ=σ, τ=τ, c=-c))

    return -c


def _pair_constants(particles, v0, pair_fdc, pair_fem):
    c11 = 0
    c1111 = 0
    c1122 = 0
    c1212 = 0
    for m in particles:
        pair_conf = Pair(m, np.array([0, 0]), pair_fem, pair_fdc)
        c11 += c_αβ_m(pair_conf, 1, 1)
        c1111 += c_αβστ_m(pair_conf, 1, 1, 1, 1)
        c1122 += c_αβστ_m(pair_conf, 1, 1, 2, 2)
        c1212 += c_αβστ_m(pair_conf, 1, 2, 1, 2)
    logging.info(
        'Pair interaction C11={0}, C1111={1}, C1122={2}, C1212={3}'.format(c11 / v0, c1111 / v0, c1122 / v0, c1212 / v0)
    )
    return c11 / v0, c1111 / v0, c1122 / v0, c1212 / v0


def _three_body_constants(pairs, v0, fem, pair_fem, fdc, pair_fdc):
    tri_c11 = 0
    tri_c1111 = 0
    tri_c1122 = 0
    tri_c1212 = 0
    for m, n in pairs:
        print('m =', m, '; n =', n)

        conf = Triplet(m, n, fem, pair_fem, fdc, pair_fdc)

        tri_c11 += c_αβ_mn(conf, 1, 1)
        tri_c1111 += c_αβστ_mn(conf, 1, 1, 1, 1)
        tri_c1122 += c_αβστ_mn(conf, 1, 1, 2, 2)
        tri_c1212 += c_αβστ_mn(conf, 1, 2, 1, 2)
        if math.fabs(tri_c1212) > math.fabs(tri_c1122):
            print('WARNING m =', m, 'n =', n, '1212 =', tri_c1212, '1122 =', tri_c1122)
    logging.info(
        'Triplet interaction C11={0}, C1111={1}, C1122={2}, C1212={3}'.format(
            tri_c11 / v0, tri_c1111 / v0, tri_c1122 / v0, tri_c1212 / v0
        )
    )
    return tri_c11 / v0, tri_c1111 / v0, tri_c1122 / v0, tri_c1212 / v0
