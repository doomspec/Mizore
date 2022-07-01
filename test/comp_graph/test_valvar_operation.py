import time
from mizore.comp_graph.comp_param import CompParam
from mizore.comp_graph.valvar import ValVar
from jax import numpy as jnp
import numpy as np

def get_mean_var(sampled):
    sampled = np.array(sampled)
    mean = sampled.mean(axis=0)
    var = sampled.var(axis=0)
    return mean, var

def operation_test(v1: ValVar, v2: ValVar, op, n_repeat=10000, seed=0):
    v3: ValVar = op(v1, v2)
    mean = v3.mean.value()
    var = v3.var.value()
    sampled = []
    for i in range(n_repeat):
        v1_sample = CompParam(val=v1.sample_gaussian(i+seed))
        v2_sample = CompParam(val=v2.sample_gaussian(i+seed*2))
        v3_sample = op(v1_sample, v2_sample)
        sampled.append(v3_sample.value())
    mean_obs, var_obs = get_mean_var(sampled)
    return (mean_obs-mean)/mean, (var_obs-var)/var

def test_division():
    y1 = ValVar(1.0, 0.1)
    y2 = ValVar(10.0, 0.1)
    mean_diff, var_diff = operation_test(y1, y2, lambda a1, a2: a1/a2, n_repeat=5000, seed=int(time.time()))
    print(mean_diff, var_diff)
    assert abs(mean_diff) < 0.01
    assert abs(var_diff) < 0.1

def test_sqrt():
    y1 = ValVar(10.0, 0.1)
    y2 = ValVar(10.0, 0.1)
    mean_diff, var_diff = operation_test(y1, y2, lambda a1, a2: (jnp.sqrt|a1)/a2, n_repeat=5000, seed=int(time.time()))
    print(mean_diff, var_diff)
    assert abs(mean_diff) < 0.01
    assert abs(var_diff) < 0.1

def test_pow():
    y1 = ValVar(10.0, 0.1)
    y2 = ValVar(10.0, 0.1)
    mean_diff, var_diff = operation_test(y1, y2, lambda a1, a2: ((lambda x: jnp.power(x,2)) | a1)/a2, n_repeat=5000, seed=int(time.time()))
    print(mean_diff, var_diff)
    assert abs(mean_diff) < 0.01
    assert abs(var_diff) < 0.1