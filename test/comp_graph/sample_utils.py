from mizore.comp_graph.value import Value
import numpy as np


def get_mean_var(sampled):
    sampled = np.array(sampled)
    mean = sampled.mean(axis=0)
    var = sampled.var(axis=0)
    return mean, var


def operation_test_binary(v1: Value, v2: Value, op, n_repeat=10000, seed=0, get_value=False):
    v3: Value = Value.binary_operator(v1, v2, op)
    mean = v3.value()
    var = v3.var.value()
    sampled = []
    for i in range(n_repeat):
        v1_sample = v1.sample_gaussian(i + seed)
        v2_sample = v2.sample_gaussian(i + seed * 2)
        v3_sample = op(v1_sample, v2_sample)
        sampled.append(v3_sample)
    mean_obs, var_obs = get_mean_var(sampled)
    if not get_value:
        return (mean_obs - mean) / mean, (var_obs - var) / var
    else:
        return mean_obs, var_obs

def operation_test_unary(v1: Value, op, n_repeat=10000, seed=0, get_value=False):
    v3: Value = op(v1)
    mean = v3.value()
    var = v3.var.value()
    sampled = []
    for i in range(n_repeat):
        v1_sample = Value(val=v1.sample_gaussian(i + seed))
        v3_sample = op(v1_sample)
        sampled.append(v3_sample.value())
    mean_obs, var_obs = get_mean_var(sampled)
    if not get_value:
        return (mean_obs - mean) / mean, (var_obs - var) / var
    else:
        return mean_obs, var_obs
