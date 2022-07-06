from circuit_utils.sample_pqc_node import simple_large_pqc_node2
from numpy import array
from numpy.testing import assert_array_almost_equal

from meta_circuit.gradient.finite_diff_gradient import finite_diff_gradient
from mizore.meta_circuit.meta_circuit import MetaCircuit


def param_shift_rule_gradient(circuit: MetaCircuit, obs_list):
    param = [0.0] * circuit.n_param
    grads = []
    for i in range(circuit.n_param):
        coeff_and_circuit = circuit.get_gradient_circuits(i)
        grads_i = []
        for obs in obs_list:
            g = 0
            for coeff, gbc in coeff_and_circuit:
                g += coeff * gbc.get_expectation_value(obs, param)
            grads_i.append(g)
        grads.append(grads_i)
    grads = array(grads)
    return grads


def test_rotation():
    node = simple_large_pqc_node2()
    grads_finite = finite_diff_gradient(node.circuit, node.obs_list)
    grads_shift_rule = param_shift_rule_gradient(node.circuit, node.obs_list)
    assert_array_almost_equal(grads_finite, grads_shift_rule)
