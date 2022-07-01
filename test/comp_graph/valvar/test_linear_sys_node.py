import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from mizore.comp_graph.valvar import ValVar
from mizore.comp_graph.calc_node.linear_sys_node import solve_linear_system, get_linear_system_partials



def test_diffs():
    A_mat_arr = [[1.0, 0.1, 0.1], [0.0, 1.0, 0.1], [0.1, 0.2, 1.0]]
    A_mat_mean = jnp.array(A_mat_arr, dtype=jnp.float64)
    A_mat_var = jnp.ones((3, 3)) * 0.01

    b_vec_arr = [1.2, -1.3, 1.4]
    b_vec_mean = jnp.array(b_vec_arr, dtype=jnp.float64)
    b_vec_var = jnp.array([0.01] * 3, dtype=jnp.float64)

    A_mat = ValVar(A_mat_mean, A_mat_var)
    b_vec = ValVar(b_vec_mean, b_vec_var)
    x_on_mean, x_partial_A, x_double_partial_A, x_partial_b = get_linear_system_partials(A_mat, b_vec)
    eps = 1e-4

    diff2_counter = 0
    for i1 in range(len(x_on_mean)):
        for i2 in range(len(x_on_mean)):
            A_mat_arr[i1][i2] -= eps
            A_mat0 = jnp.array(A_mat_arr, dtype=jnp.float64)
            x_on_mean1_ = jnp.linalg.pinv(A_mat0)@b_vec_mean
            A_mat_arr[i1][i2] += 2*eps
            A_mat1 = jnp.array(A_mat_arr, dtype=jnp.float64)
            x_on_mean1 = jnp.linalg.pinv(A_mat1)@b_vec_mean
            diff1 = (x_on_mean1-x_on_mean1_)/(2*eps)
            assert jnp.linalg.norm(x_partial_A[i1][i2]-diff1) < 1e-3

            # It is hard to test the second derivative
            # The finite difference is very unstable

            diff2 = (x_on_mean1+x_on_mean1_-2*x_on_mean)/(eps**2)
            if jnp.linalg.norm(x_double_partial_A[i1][i2]-diff2) < 1:
                diff2_counter += 1

    for i1 in range(len(x_on_mean)):
        A_inv = jnp.linalg.pinv(A_mat_mean)
        b_vec_arr[i1] -= eps
        x_on_mean1_ = A_inv @ jnp.array(b_vec_arr, dtype=jnp.float64)
        b_vec_arr[i1] += 2 * eps
        x_on_mean1 = A_inv @ jnp.array(b_vec_arr, dtype=jnp.float64)
        diff1 = (x_on_mean1 - x_on_mean1_) / (2 * eps)
        assert jnp.linalg.norm(x_partial_b[i1] - diff1) < 1e-3
    #print(diff2_counter)