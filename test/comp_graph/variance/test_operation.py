import time
from mizore.comp_graph.value import Value, Variable
from jax import numpy as jnp
from sample_utils import operation_test_binary, operation_test_unary


def test_plus():
    y1 = Variable(10.0, 0.1)
    mean, var = operation_test_unary(y1, lambda a1: a1 + a1, n_repeat=10000, seed=int(time.time()), get_value=True)
    mean2, var2 = operation_test_unary(y1, lambda a1: 2 * a1, n_repeat=10000, seed=int(time.time()), get_value=True)
    assert abs(mean - 20.0) < 0.1
    assert abs(mean2 - 20.0) < 0.1
    assert abs(var - 0.4) < 0.05
    assert abs(var2 - 0.4) < 0.05


def test_division():
    y1 = Variable(1.0, 0.1)
    y2 = Variable(10.0, 0.1)
    mean_diff, var_diff = operation_test_binary(y1, y2, lambda a1, a2: a1 / a2, n_repeat=10000, seed=int(time.time()))
    print(mean_diff, var_diff)
    assert abs(mean_diff) < 0.01
    assert abs(var_diff) < 0.1


def test_sqrt():
    y1 = Variable(10.0, 0.1)
    y2 = Variable(10.0, 0.1)
    t1 = time.time()
    mean_diff, var_diff = operation_test_binary(y1, y2, lambda a1, a2: jnp.sqrt(a1)/ a2, n_repeat=10000,
                                                seed=int(time.time()))
    print(time.time() - t1)
    print(mean_diff, var_diff)
    assert abs(mean_diff) < 0.01
    assert abs(var_diff) < 0.1


def test_pow():
    y1 = Variable(10.0, 0.1)
    y2 = Variable(10.0, 0.1)
    mean_diff, var_diff = operation_test_binary(y1, y2, lambda a1, a2: (a1 ** 2) / a2, n_repeat=10000, seed=int(time.time()))
    print(mean_diff, var_diff)
    assert abs(mean_diff) < 0.01
    assert abs(var_diff) < 0.1

def lstsq(A, b):
    return jnp.linalg.lstsq(A, b)[0]

def test_lstsq():
    A_mat_arr = [[2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
    A_mat_mean = jnp.array(A_mat_arr, dtype=jnp.float64)
    A_mat_var = jnp.ones((3, 3)) * 0.01

    b_vec_arr = [1.2, -1.3, 1.4]
    b_vec_mean = jnp.array(b_vec_arr, dtype=jnp.float64)
    b_vec_var = jnp.array([0.01] * 3, dtype=jnp.float64)

    A_mat = Variable(A_mat_mean, A_mat_var)
    b_vec = Variable(b_vec_mean, b_vec_var)

    mean_diff, var_diff = operation_test_binary(A_mat, b_vec, lstsq, n_repeat=50000, seed=int(time.time()))

    print(mean_diff, var_diff)
    assert max(mean_diff) < 0.01
    assert 0.0 < max(var_diff) < 1.0  # It seems like first order method can greatly underestimate the variance
