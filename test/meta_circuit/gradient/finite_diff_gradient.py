from mizore.meta_circuit.meta_circuit import MetaCircuit
from numpy import array


def finite_diff_gradient(circuit: MetaCircuit, obs, eps=1e-6):
    param = [0.0] * circuit.n_param
    expv0 = circuit.get_expectation_value(obs, param)
    expv = []
    for i in range(circuit.n_param):
        param[i] += eps
        expv_i = circuit.get_expectation_value(obs, param)
        expv.append(expv_i)
        param[i] -= eps
    expv = array(expv)
    grad_finite = (expv - expv0) / eps
    return grad_finite
