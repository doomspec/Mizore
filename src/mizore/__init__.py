import jax
from jax.numpy import array as jax_array
from numpy import array as np_array

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)  # x64 precision is enabled by default in Mizore


def to_jax_array(array) -> jax.numpy.ndarray:
    return jax_array(array, copy=False)


def to_np_array(array):
    return np_array(array, copy=False)
