# Elastic Const
Computation of crystal elastic constants using effective pair and three
body potentials. Pair and three body potentials can be represented as
explicit or implicit functions. In the latter case potential energy is
calculated using FEM simulation.

The main module is `elastic_const.elastic_const_computation`. It offers
two functions to compute elastic constants `compute_constants_xy_method`
and `compute_constants_r_method`. They differ by how the consider
potential functions. In the `compute_constants_xy_method` potential
function is represented as a function which accepts x and y coordinates
of the particles. And `compute_constants_r_method` considers potentials
as functions which accept distances between particles. Actual
implementations of these methods are in the modules
`elastic_const.xy_method` and `elastic_const.r_method`.

Module `elastic_const.xy_method` is designed to work with classes
`PairFemSimulation`, `TripletFemSimulation`,
`PairForceDerivativeComputation`, `TripletForceDerivativeComputation`,
which are wrappers for external FEM simulation program. Module
`r_method` is designed to work with generic two and three body
potentials.