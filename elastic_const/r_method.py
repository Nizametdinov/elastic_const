import numpy as np
import logging

from elastic_const.misc import pairwise_distances


def pair_constants(particles, v0, potential2_func):
    c11 = 0
    c1111 = 0
    c1122 = 0
    c1212 = 0
    for m in particles:
        f2 = potential2_func(np.linalg.norm(m))
        logging.debug('r = {r}; df_2/dr^2 = {df2}; d^2f_2/d(r^2)^2 = {d2f2}'.format(
            df2=f2.first_derivative(), d2f2=f2.second_derivative(), r=f2.r))

        c11 += c_αβ_m(f2, m, 1, 1) / v0
        c1111 += c_αβστ_m(f2, m, 1, 1, 1, 1) / v0
        c1122 += c_αβστ_m(f2, m, 1, 1, 2, 2) / v0
        c1212 += c_αβστ_m(f2, m, 1, 2, 1, 2) / v0
    logging.info(
        'Pair interaction C11={0}, C1111={1}, C1122={2}, C1212={3}'.format(c11, c1111, c1122, c1212)
    )
    return c11, c1111, c1122, c1212


def three_body_constants(pairs, v0, potential3_func):
    tri_c11 = 0
    tri_c1111 = 0
    tri_c1122 = 0
    tri_c1212 = 0
    for m, n in pairs:
        print('m =', m, '; n =', n)
        r12, r23, r13 = pairwise_distances([np.zeros_like(m), m, n])
        f3 = potential3_func(r12, r23, r13)
        logging.debug('m = {m}; n = {n}'.format(m=m, n=n))
        logging.debug(f3.first_derivatives())
        logging.debug(f3.second_derivatives())

        tri_c11 += c_αβ_mn(f3, m, n, 1, 1) / v0
        tri_c1111 += c_αβστ_mn(f3, m, n, 1, 1, 1, 1) / v0
        tri_c1122 += c_αβστ_mn(f3, m, n, 1, 1, 2, 2) / v0
        tri_c1212 += c_αβστ_mn(f3, m, n, 1, 2, 1, 2) / v0

    logging.info(
        'Triplet interaction C11={0}, C1111={1}, C1122={2}, C1212={3}'.format(tri_c11, tri_c1111, tri_c1122, tri_c1212)
    )
    return tri_c11, tri_c1111, tri_c1122, tri_c1212


def c_αβ_m(f2, m, α, β):
    return 2 * m[α - 1] * m[β - 1] * f2.first_derivative()


def c_αβστ_m(f2, m, α, β, σ, τ):
    return 4 * m[α - 1] * m[β - 1] * m[σ - 1] * m[τ - 1] * f2.second_derivative()


def c_αβ_mn(f3, m, n, α, β):
    x12, x13, x23 = m, n, n - m
    df3 = f3.first_derivatives()
    c  = 2 * x12[α - 1] * x12[β - 1] * df3.dr12
    c += 2 * x13[α - 1] * x13[β - 1] * df3.dr13
    c += 2 * x23[α - 1] * x23[β - 1] * df3.dr23
    return c


def c_αβστ_mn(f3, m, n, α, β, σ, τ):
    # TODO: направление вектора n - m
    x12, x13, x23 = m, n, n - m
    d2f3 = f3.second_derivatives()
    c  = 4 * x12[α - 1] * x12[β - 1] * x12[σ - 1] * x12[τ - 1] * d2f3.dr12r12
    c += 4 * x13[α - 1] * x13[β - 1] * x13[σ - 1] * x13[τ - 1] * d2f3.dr13r13
    c += 4 * x23[α - 1] * x23[β - 1] * x23[σ - 1] * x23[τ - 1] * d2f3.dr23r23
    c += 8 * x12[α - 1] * x12[β - 1] * x23[σ - 1] * x23[τ - 1] * d2f3.dr12r23
    c += 8 * x13[α - 1] * x13[β - 1] * x23[σ - 1] * x23[τ - 1] * d2f3.dr13r23
    c += 8 * x12[α - 1] * x12[β - 1] * x13[σ - 1] * x13[τ - 1] * d2f3.dr12r13
    logging.debug('m = {m}; n = {n}; ΔC{α}{β}{σ}{τ} = {c}'.format(m=m, n=n, α=α, β=β, σ=σ, τ=τ, c=c))
    return c
