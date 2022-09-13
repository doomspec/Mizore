from math import ceil

from chemistry.simple_mols import simple_4_qubit_lih
from circuit_utils.sample_circuit import circuit_for_test_0
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.measurement.l1 import L1Sampling, L1Allocation
from jax import numpy as jnp
from numpy.testing import assert_allclose


def test_compare_separate_and_together():
    shot_num = 100
    hamil, _ = simple_4_qubit_lih().remove_constant()
    hamil_weight = hamil.get_l1_norm_omit_const()
    circuit = circuit_for_test_0()
    node0 = DeviceCircuitNode(circuit, hamil)
    node0.shot_num.set_value(shot_num)
    CircuitRunner() | node0
    L1Sampling(state_ignorant=True) | node0
    expv0 = node0()
    mean0 = expv0.value()
    var0 = expv0.var.value()
    expv_by_sum = Value(0.0)
    n_shots = 0
    for qset_op_weight in hamil.qset_op_weight_omit_const():
        node = DeviceCircuitNode(circuit, QubitOperator.from_qset_op_weight(*qset_op_weight))
        expv_by_sum += node.expv
        node.shot_num.set_value(abs(qset_op_weight[2])/hamil_weight*shot_num)
        n_shots += abs(qset_op_weight[2])/hamil_weight*shot_num
        CircuitRunner() | node
        L1Sampling(state_ignorant=True) | node
    print(n_shots)
    mean_by_sum = expv_by_sum.value()
    var_by_sum = expv_by_sum.var.value()
    assert_allclose(mean_by_sum, mean0)
    assert var0 > var_by_sum  # To ensure L1Sampling always underestimate variance
    assert_allclose(var_by_sum, var0, rtol=0.005, atol=0.0002)
    print(var_by_sum, var0)


def test_compare_separate_and_together1():
    shot_num = 100
    hamil, _ = simple_4_qubit_lih().remove_constant()
    hamil_weight = hamil.get_l1_norm_omit_const()
    circuit = circuit_for_test_0()
    node0 = DeviceCircuitNode(circuit, hamil)
    node0.shot_num.set_value(shot_num)
    L1Allocation.prepare() | node0
    CircuitRunner() | node0
    L1Allocation() | node0
    expv0 = node0()
    mean0 = expv0.value()
    var0 = expv0.var.value()
    expv_by_sum = Value(0.0)
    n_shots = 0
    for qset_op_weight in hamil.qset_op_weight_omit_const():
        node = DeviceCircuitNode(circuit, QubitOperator.from_qset_op_weight(*qset_op_weight))
        expv_by_sum += node.expv
        node.shot_num.set_value(abs(qset_op_weight[2])/hamil_weight*shot_num)
        n_shots += abs(qset_op_weight[2])/hamil_weight*shot_num
        CircuitRunner() | node
        L1Sampling(state_ignorant=False) | node
    print(n_shots)
    mean_by_sum = expv_by_sum.value()
    var_by_sum = expv_by_sum.var.value()
    assert_allclose(mean_by_sum, mean0)
    assert var0 < var_by_sum
    assert_allclose(var_by_sum, var0, rtol=0.005, atol=0.0002)
    print(var_by_sum, var0)
