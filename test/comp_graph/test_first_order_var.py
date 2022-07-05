import jax.numpy as jnp

from mizore.comp_graph.value import Value

from numpy.testing import assert_array_almost_equal


def double_trace(mat):
    return jnp.array([jnp.trace(mat), 2.0 * jnp.trace(mat)])


def test_mat():
    A = Value(jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    A.set_to_random_variable(jnp.ones((3, 3)) * 0.01)
    At = (double_trace | A)
    At.const_approx = True
    assert_array_almost_equal(At.value(), [2., 4.])
    assert_array_almost_equal((At*2).var.value(), [0.12, 0.48])


def test_unary():
    a = Value.random_variable(1.0, 0.2)
    b = Value.unary_operator(a, jnp.sqrt) + a
    assert_array_almost_equal(b.var.value(), 0.45)
    assert_array_almost_equal(b.value(), 2.0)


def lstsq(A, b):
    return jnp.linalg.lstsq(A, b)[0]


def test_binary():
    A = Value.random_variable(jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), jnp.ones((3, 3)) * 0.01)
    B = Value.random_variable(jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), jnp.ones((3, 3)) * 0.01)
    b = Value.random_variable(jnp.array([1.0, 0.0, 0.0]), jnp.ones((3,)) * 0.01)
    x = Value.binary_operator(A, b, lstsq)
    C = Value.binary_operator(A, B, jnp.matmul)

    assert_array_almost_equal(C.value(), [[1., 2., 0.], [0., 1., 0.], [0., 0., 0.]])
    assert_array_almost_equal(C.var.value(), [[0.03, 0.04, 0.02], [0.02, 0.03, 0.01], [0.01, 0.02, 0.]])
    assert_array_almost_equal(x.value(), [9.9999994e-01, -2.5211193e-09, 0.0000000e+00])
    assert_array_almost_equal(x.var.value(), [0.03999999, 0.02, 0.02])
