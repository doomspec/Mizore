from circuit_utils.sample_pqc_node import simple_large_pqc_node2
from numpy import array
from numpy.testing import assert_allclose

from meta_circuit.gradient.finite_diff_gradient import finite_diff_gradient
from mizore.meta_circuit.meta_circuit import MetaCircuit


def param_shift_rule_gradient(circuit: MetaCircuit, obs):
    param = [0.0] * circuit.n_param
    grads = []
    for i in range(circuit.n_param):
        coeff_and_circuit = circuit.get_gradient_circuits(i)
        g = 0
        for coeff, gbc in coeff_and_circuit:
            g += coeff * gbc.get_expectation_value(obs, param)
        grads.append(g)
    grads = array(grads)
    return grads


def test_rotation():
    node = simple_large_pqc_node2()
    grads_finite = finite_diff_gradient(node.circuit, node.obs)
    grads_shift_rule = param_shift_rule_gradient(node.circuit, node.obs)
    assert_allclose(grads_finite, grads_shift_rule, atol=1e-6)
