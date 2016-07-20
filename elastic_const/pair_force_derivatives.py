from scipy.misc import derivative
from collections import namedtuple

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5


class PairForceDerivative(namedtuple('PairForceDerivative', ['distance', 'derivative', 'force', 'n'])):
    def rotate(self, to, origin):
        """
        Compute derivatives of x and y components of force acting on second particle when it has
        coordinates `to` and first particle has coordinates `origin`
        """
        if self.n != 1:
            raise NotImplementedError
        x = to[0] - origin[0]
        y = to[1] - origin[1]
        distance_sqr = self.distance * self.distance
        dFx_dx = self.derivative * x * x / distance_sqr
        dFx_dx += self.force * y * y / (distance_sqr * self.distance)
        dFy_dy = self.derivative * y * y / distance_sqr
        dFy_dy += self.force * x * x / (distance_sqr * self.distance)
        dFx_dy = self.derivative * x * y / distance_sqr
        dFx_dy -= self.force * x * y / (distance_sqr * self.distance)
        # dFx_dy == dFy_dx
        return dFx_dx, dFx_dy, dFy_dy


class PairForceDerivativeComputation(object):
    def __init__(self, simulation, order=FINITE_DIFF_ORDER, step=FINITE_DIFF_STEP, r=1., derivative_func=derivative):
        self.simulation = simulation
        self.order = order
        self.step = step
        self.r = r
        self.derivative_func = derivative_func

    def derivative_of_force(self, distance, n=1):
        def force_func(arg):
            return self.simulation.compute_forces(arg).force

        force = self.simulation.compute_forces(distance).force
        step = self.step * (distance - 2 * self.r) if distance - 2 * self.r < 1.0 else self.step
        dF_dr = self.derivative_func(force_func, distance, dx=step, order=self.order, n=n)
        result = PairForceDerivative(distance, dF_dr, force, n=n)
        return result
