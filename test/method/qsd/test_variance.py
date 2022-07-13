from mizore.comp_graph.value import Variable, Value
from mizore.method.qsd.generalized_eigv import generalized_eigv_by_wang
from mizore.testing.sample_utils import operation_test_binary

import jax

jax.config.update("jax_enable_x64", True)

def make_krylov_single_ref_random_mat(mean_mat, var):
    n_dim = len(mean_mat)
    rand_mat = [[None for _ in range(n_dim)] for __ in range(n_dim)]
    for i in range(n_dim):
        rand_mat[i][i] = Variable(mean_mat[i][i].real, var * 10)
        assert abs(mean_mat[i][i].imag) < 1e-11
    for j in range(1, n_dim):
        rand_mat[0][j] = Variable(mean_mat[0][j].real, var) + Variable(mean_mat[0][j].imag, var) * 1j
        rand_mat[j][0] = Value.conjugate(rand_mat[0][j])
    for i in range(1, n_dim):
        for j in range(i+1, n_dim):
            rand_mat[i][j] = rand_mat[0][j-i]
            rand_mat[j][i] = rand_mat[j-i][0]
    rand_mat = Value.matrix(rand_mat)
    return rand_mat


def test_eigh_var():
    S_mat_arr = jax.numpy.array([[1. + 0.j, 0.98157393 - 0.180359j, 0.92710746 - 0.35310831j],
                                 [0.98157393 + 0.180359j, 1. + 0.j, 0.98157393 - 0.180359j],
                                 [0.92710746 + 0.35310831j, 0.98157393 + 0.180359j, 1. + 0.j]])
    H_mat_arr = jax.numpy.array([[-0.9081943 + 0.00000000e+00j, -0.88902548 + 1.83581115e-01j,
                                  -0.83238507 + 3.59075953e-01j],
                                 [-0.88902548 - 1.83581115e-01j, -0.9081943 - 3.29792278e-18j,
                                  -0.88902548 + 1.83581115e-01j],
                                 [-0.83238507 - 3.59075953e-01j, -0.88902548 - 1.83581115e-01j,
                                  -0.9081943 - 1.30392252e-17j]])
    var = 1e-14

    H_mat = make_krylov_single_ref_random_mat(H_mat_arr, var)
    S_mat = make_krylov_single_ref_random_mat(S_mat_arr, 0.0)

    def generalized_eigvals(H_, S_):
        return generalized_eigv_by_wang(H_, S_, eigvals_only=True, eps=1e-10)  # + 1.0752071979378575

    means, vars = operation_test_binary(H_mat, S_mat, generalized_eigvals, n_repeat=100000, use_jit=True)

    print(means)
    print(means[0] - means[1])
    print(vars)
    print(vars[0] - vars[1])
