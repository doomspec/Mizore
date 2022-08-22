from chemistry.simple_mols import simple_4_qubit_lih
from circuit_utils.sample_circuit import circuit_for_test_0
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.measurement.l1sampling import L1Sampling
from jax import numpy as jnp
from numpy.testing import assert_allclose


def test_compare_separate_and_together():
    shot_num = 1000

    hamil = simple_4_qubit_lih()
    circuit = circuit_for_test_0()
    node0 = DeviceCircuitNode(circuit, hamil)
    node0.shot_num.set_value(shot_num)
    CircuitRunner() | node0
    L1Sampling() | node0
    expv0 = node0()
    mean0 = expv0.value()
    var0 = expv0.var.value()

    obs = []
    shot_allocate = []
    for qset_op_weight in hamil.qset_op_weight():
        obs.append(QubitOperator.from_qset_op_weight(*qset_op_weight))
        qset = qset_op_weight[0]
        if len(qset) == 0:
            shot_allocate.append(0.000000001)
        else:
            shot_allocate.append(abs(qset_op_weight[2]))
    shot_allocate = jnp.array(shot_allocate)
    shot_allocate /= jnp.sum(shot_allocate)
    #shot_allocate *= shot_num
    #shot_allocate = jnp.ceil(shot_allocate)

    node1 = DeviceCircuitNode(circuit, obs)

    CircuitRunner() | node1
    L1Sampling(shot_allocate={node1: shot_allocate}) | node1

    expv1 = node1.expv_vector()
    expv1_sum = Value.unary_operator(expv1, jnp.sum)
    mean1 = expv1_sum.value()
    var1 = expv1_sum.var.value()
    assert_allclose(mean1, mean0)
    assert var0 < var1  # To ensure L1Sampling always underestimate variance
    assert_allclose(var1, var0, rtol=0.005, atol=0.0002)
    print(var1, var0)
