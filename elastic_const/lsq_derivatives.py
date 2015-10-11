import numpy as np


def lsq_derivative(func, x, dx, order=5, poly_order=3):
    """Calculation of first derivative with polynomial approximation"""
    assert order % 2 == 1
    if order == 11:
        return lsq_derivative11(func, x, dx, poly_order)
    xs = (np.linspace(0, order, order) - (order - 1) / 2) * dx
    ys = np.array([func(x + dx) for dx in xs])
    fit = np.polyfit(xs, ys, poly_order)
    return fit[-2]


def lsq_derivative11(func, x, dx, poly_order=5):
    assert 1 <= poly_order <= 10
    if poly_order == 1 or poly_order == 2:
        coeff = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) / 110.
    elif poly_order == 3 or poly_order == 4:
        coeff = np.array([300, -294, -532, -503, -296, 0, 296, 503, 532, 294, -300]) / 5148.
    elif poly_order == 5 or poly_order == 6:
        coeff = np.array([-573, 2166, -1249, -3774, -3084, 0, 3084, 3774, 1249, -2166, 573]) / 17160.
    elif poly_order == 7 or poly_order == 8:
        coeff = np.array([18588, -141411, 424084, -483816, -852936, 0, 852936, 483816, -424084, 141411, -18588]) / \
                2042040.
    elif poly_order == 9 or poly_order == 10:
        coeff = np.array([-2, 25, -150, 600, -2100, 0, 2100, -600, 150, -25, 2]) / 2520.

    ys = np.array([func(x + i * dx) for i in range(-5, 0)] + [0] + [func(x + i * dx) for i in range(1, 6)])
    return coeff.dot(ys)