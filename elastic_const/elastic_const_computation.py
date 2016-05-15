import logging
import traceback
from elastic_const import xy_method
from elastic_const import r_method
from elastic_const.derivatives_with_distance import Potential2, Potential3
from elastic_const.crystals import Lattice


def compute_constants_xy_method(max_order, triplet_fem, pair_fem, triplet_fdc, pair_fdc, lattice: Lattice):
    def pair_const_func(particles, v0):
        return xy_method.pair_constants(particles, v0, pair_fem, pair_fdc)

    def triplet_const_func(pairs, v0):
        return xy_method.three_body_constants(pairs, v0, triplet_fem, pair_fem, triplet_fdc, pair_fdc)

    return compute_constants_incremental(max_order, pair_const_func, triplet_const_func, lattice)


def compute_constants_r_method(max_order, f2dc, f3dc, lattice: Lattice):
    def potential2(r):
        return Potential2(f2dc, r)

    def potential3(r12, r23, r13):
        return Potential3(f3dc, r12, r23, r13)

    def pair_const_func(particles, v0):
        return r_method.pair_constants(particles, v0, potential2)

    def triplet_const_func(pairs, v0):
        return r_method.three_body_constants(pairs, v0, potential3)

    return compute_constants_incremental(max_order, pair_const_func, triplet_const_func, lattice)


def compute_constants_incremental(max_order, pair_const_func, triplet_const_func, lattice: Lattice):
    """
    Incremental computation of elastic constants for 2d crystal
    """
    try:
        v0 = lattice.ws_cell_volume()
        c11, c1111, c1122, c1212 = 0, 0, 0, 0
        tri_c11, tri_c1111, tri_c1122, tri_c1212 = 0, 0, 0, 0
        for order in range(max_order):
            print('order =', order + 1)
            d_c11, d_c1111, d_c1122, d_c1212 = pair_const_func(lattice.points_for(order), v0)
            c11 += d_c11 / 2
            c1111 += d_c1111 / 2
            c1122 += d_c1122 / 2
            c1212 += d_c1212 / 2

            d_tri_c11, d_tri_c1111, d_tri_c1122, d_tri_c1212 = triplet_const_func(lattice.pairs_for(order), v0)
            tri_c11 += d_tri_c11 / 3
            tri_c1111 += d_tri_c1111 / 3
            tri_c1122 += d_tri_c1122 / 3
            tri_c1212 += d_tri_c1212 / 3

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
