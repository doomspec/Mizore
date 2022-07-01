import jax.numpy as jnp
from jax.config import config

from mizore.comp_graph.node.calc_node import CalcNode

config.update("jax_enable_x64", True)
from mizore.comp_graph.valvar import ValVar
from mizore.comp_graph.calc_node.linear_sys_node import solve_linear_system, get_linear_system_partials, LinearSysNode

A_mat_arr = [[1.0, 0.1, 0.1], [0.0, 1.0, 0.1], [0.1, 0.2, 1.0]]
A_mat_mean = jnp.array(A_mat_arr, dtype=jnp.float64)
A_mat_var = jnp.ones((3, 3)) * 0.01

b_vec_arr = [1.2, -1.3, 1.4]
b_vec_mean = jnp.array(b_vec_arr, dtype=jnp.float64)
b_vec_var = jnp.array([0.01] * 3, dtype=jnp.float64)

A_mat = ValVar(A_mat_mean, A_mat_var)
b_vec = ValVar(b_vec_mean, b_vec_var)


node = LinearSysNode(A_mat, b_vec)
node().show_value()