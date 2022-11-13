from typing import Tuple
from mizore.comp_graph.value import Value
import numpy as np
from numpy import ndarray


def get_mean_var(sampled):
    sampled = np.array(sampled)
    mean = sampled.mean(axis=0)
    var = sampled.var(axis=0)
    return mean, var


def operation_test_binary(v1: Value, v2: Value, op, n_repeat=10000, use_jit=False) -> Tuple[
    Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
    """
    :return: a tuple for mean and a tuple for variance. In each tuple, the first element is from sampling and the
                second element is from auto-variance
    """
    v3: Value = Value.binary_operator(v1, v2, op)
    mean = v3.value()
    var = v3.var.value()
    sampled = []
    v1_sampled = list(v1.gaussian_sample_iter(n_repeat, use_jit=use_jit))
    v2_sampled = list(v2.gaussian_sample_iter(n_repeat, use_jit=use_jit))

    for i in range(n_repeat):
        v1_sample = v1_sampled[i]
        v2_sample = v2_sampled[i]
        v3_sample = op(v1_sample, v2_sample)
        sampled.append(v3_sample)
    mean_obs, var_obs = get_mean_var(sampled)

    return (mean_obs, mean), (var_obs, var)


def operation_test_unary(v1: Value, op, n_repeat=10000) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
    """
    :return: a tuple for mean and a tuple for variance. In each tuple, the first element is from sampling and the
                second element is from auto-variance
    """
    v3: Value = Value.unary_operator(v1, op)
    mean = v3.value()
    var = v3.var.value()
    sampled = []
    for v1_sample in v1.gaussian_sample_iter(n_repeat):
        v3_sample = op(v1_sample)
        sampled.append(v3_sample)
    mean_obs, var_obs = get_mean_var(sampled)
    return (mean_obs, mean), (var_obs, var)
