import time
from mizore.comp_graph.value import Variable
from jax import numpy as jnp
from mizore.testing.sample_utils import operation_test_binary, operation_test_unary
from numpy.testing import assert_allclose


def test_plus():
    y1 = Variable(10.0, 0.1)
    means0, vars0 = operation_test_unary(y1, lambda a1: a1 + a1, n_repeat=10000, seed=int(time.time()))
    means1, vars1 = operation_test_unary(y1, lambda a1: 2 * a1, n_repeat=10000, seed=int(time.time()))
    assert_allclose(*means0, rtol=0.1)
    assert_allclose(*means1, rtol=0.1)
    assert_allclose(*means0, rtol=0.1)
    assert_allclose(*vars1, rtol=0.1)


def test_division():
    y1 = Variable(1.0, 0.1)
    y2 = Variable(10.0, 0.1)
    means, vars = operation_test_binary(y1, y2, lambda a1, a2: a1 / a2, n_repeat=10000, seed=int(time.time()))
    assert_allclose(*means, rtol=0.1)
    assert_allclose(*vars, rtol=0.1)


def test_sqrt():
    y1 = Variable(10.0, 0.1)
    y2 = Variable(10.0, 0.1)
    t1 = time.time()
    means, vars = operation_test_binary(y1, y2, lambda a1, a2: jnp.sqrt(a1)/ a2, n_repeat=10000,
                                                seed=int(time.time()))
    assert_allclose(*means, rtol=0.1)
    assert_allclose(*vars, rtol=0.1)


def test_pow():
    y1 = Variable(10.0, 0.1)
    y2 = Variable(10.0, 0.1)
    means, vars = operation_test_binary(y1, y2, lambda a1, a2: (a1 ** 2) / a2, n_repeat=10000, seed=int(time.time()))
    assert_allclose(*means, rtol=0.1)
    assert_allclose(*vars, rtol=0.1)

def lstsq(A, b):
    return jnp.linalg.lstsq(A, b)[0]

def test_lstsq():
    A_mat_arr = [[2.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.0, 1.0, 2.0]]
    A_mat_mean = jnp.array(A_mat_arr, dtype=jnp.float64)
    A_mat_var = jnp.ones((3, 3)) * 0.01

    b_vec_arr = [1.2, -1.3, 1.4]
    b_vec_mean = jnp.array(b_vec_arr, dtype=jnp.float64)
    b_vec_var = jnp.array([0.01] * 3, dtype=jnp.float64)

    A_mat = Variable(A_mat_mean, A_mat_var)
    b_vec = Variable(b_vec_mean, b_vec_var)

    means, vars = operation_test_binary(A_mat, b_vec, lstsq, n_repeat=100000, seed=int(time.time()))


    assert_allclose(*means, rtol=0.1)
    assert_allclose(*vars, rtol=0.1)
    #assert vars[0]-vars[1]  # It seems like first order method can greatly underestimate the variance
