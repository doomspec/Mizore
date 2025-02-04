from jax.numpy import array as jax_array
from numpy import array as np_array


def to_jax_array(array):
    return jax_array(array, copy=False)


def to_np_array(array):
    return np_array(array, copy=False)
