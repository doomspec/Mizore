import time
from math import pi
import numpy as np
from mizore.backend_circuit.backend_circuit import BackendCircuit
from mizore.backend_circuit.backend_state import BackendState
from mizore.backend_circuit.one_qubit_gates import Hadamard
from mizore.backend_circuit.rotations import SingleRotation
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.measurement.naive import NaiveMeasurement
from mizore import np_array
from numpy.testing import assert_allclose


def get_single_ob_mean_var(backend_state: BackendState, qset_op_weight, shot_num, seed=1):
    state = backend_state.copy()
    gate_list = []
    qset, op, weight = qset_op_weight
    for i in range(len(qset)):
        if op[i] == 1:
            gate_list.append(Hadamard(qset[i]))
        if op[i] == 2:
            gate_list.append(SingleRotation(1, qset[i], -pi / 2))
    basis_rotation = BackendCircuit(state.n_qubit, gate_list)
    basis_rotation.update_quantum_state(state)
    sampled_qsets = state.sample_1_qset(shot_num, seed=seed)
    observed_values = []
    for sampled_qset in sampled_qsets:
        res = 1
        for index in qset:
            if index in sampled_qset:
                res *= -1
        observed_values.append(res)
    observed_mean = np.mean(observed_values) * weight
    return observed_mean


def test_single_op():
    n_qubit = 3
    qset_op_weight = ((0, 1, 2), (3, 1, 2), 2.0)
    ob = QubitOperator.from_qset_op_weight(*qset_op_weight)
    op = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('Y1')
    blk = RotationGroup(op, fixed_angle_shift=np_array([0.1, 0.5, 1.0, 0.1]))
    circuit = MetaCircuit(n_qubit, [blk])
    node = DeviceCircuitNode(circuit, ob)
    shot_num = 1000
    node.shot_num.set_value(shot_num)
    expv = node()
    cg = expv.build_graph()
    CircuitRunner(state_processor_gens=[NaiveMeasurement(state_ignorant=False)]) | cg
    var_actual = expv.var.value()
    mean_actual = expv.value()
    print(mean_actual, var_actual)
    state = circuit.get_backend_state()
    observed_means = []
    for i in range(10000):
        observed_mean = get_single_ob_mean_var(state, qset_op_weight, shot_num, seed=int(time.time()) + i * 7)
        observed_means.append(observed_mean)
    mean_experiment = np.mean(observed_means)
    var_experiment = np.var(observed_means)
    print(mean_experiment, var_experiment)
    assert_allclose((mean_experiment, var_experiment), (mean_actual, var_actual), rtol=0.05, atol=0.0001)
