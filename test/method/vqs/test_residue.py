import time

import jax.numpy as jnp

from mizore.comp_graph.value import Variable
from mizore.testing.sample_utils import operation_test_binary

from jax.numpy.linalg import lstsq

from numpy.testing import assert_allclose


def test_residue():
    A_mat_arr = [[2.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.0, 1.0, 2.0]]
    A_mat_mean = jnp.array(A_mat_arr, dtype=jnp.float64)
    A_mat_var = jnp.ones((3, 3)) * 0.001

    b_vec_arr = [1.2, -1.3, 1.4]
    b_vec_mean = jnp.array(b_vec_arr, dtype=jnp.float64)
    b_vec_var = jnp.array([0.001] * 3, dtype=jnp.float64)

    A_mat = Variable(A_mat_mean, A_mat_var)
    b_vec = Variable(b_vec_mean, b_vec_var)

    def residue(A, b):
        return A_mat.value()@lstsq(A, b)[0]-b_vec.value()

    means, vars = operation_test_binary(A_mat, b_vec, residue, n_repeat=100000, seed=int(time.time()))


    print(vars[0]/b_vec.value())

    assert_allclose(*means, atol=0.02)
    assert_allclose(*vars, rtol=0.1)

