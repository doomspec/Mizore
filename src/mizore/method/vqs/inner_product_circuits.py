from math import pi
from typing import List, Union

from mizore.backend_circuit.multi_qubit_gates import PauliGate
from mizore.backend_circuit.one_qubit_gates import GlobalPhase, Hadamard
from mizore.backend_circuit.rotations import SingleRotation
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block.controlled import Controlled
from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator

import numpy as np


def diff_inner_product_real(circuit: MetaCircuit, index1, index2, param: Value):
    new_blocks = circuit.blocks
    n_qubit = circuit.n_qubit

    i_block1, _ = circuit.get_block_index_by_param_index(index1)
    i_block2, _ = circuit.get_block_index_by_param_index(index2)

    block1 = new_blocks[i_block1]
    block2 = new_blocks[i_block2]

    assert isinstance(block1, Rotation)
    assert isinstance(block2, Rotation)

    diff_1 = Controlled(GateGroup(PauliGate(block1.qset, block1.pauli_ops), GlobalPhase(-pi / 2)), (n_qubit,), [0])
    diff_2 = Controlled(GateGroup(PauliGate(block2.qset, block2.pauli_ops), GlobalPhase(-pi / 2)), (n_qubit,), [1])

    if i_block2 < i_block1:
        i_block1, i_block2 = i_block2, i_block1

    new_blocks = new_blocks[:i_block2 + 1]
    new_blocks.insert(i_block2 + 1, diff_2)
    new_blocks.insert(i_block1 + 1, diff_1)
    new_blocks.insert(0, GateGroup(Hadamard(n_qubit)))

    new_circuit = MetaCircuit(circuit.n_qubit + 1, new_blocks)
    obs = QubitOperator(f"X{n_qubit}")
    node = DeviceCircuitNode(new_circuit, obs, name="DiffInnerProd", param=param)

    expv = node()

    return expv * 0.25 * block1.weight * block2.weight


def A_mat_real(circuit: MetaCircuit, param: Value):
    n_param = circuit.n_param
    A: List[List[Union[Value, None]]] = [[None for _ in range(n_param)] for _ in range(n_param)]
    for i in range(n_param):
        i_block1, _ = circuit.get_block_index_by_param_index(i)
        weight = circuit._blocks[i_block1].weight
        A[i][i] = Value(0.25 * (weight ** 2))  # Only work for exp(-i/2 \theta * weight)
        for j in range(i + 1, n_param):
            A[i][j] = diff_inner_product_real(circuit, i, j, param)
            A[j][i] = A[i][j]  # .conjugate()
        A[i] = Value.array(A[i])
    A = Value.array(A)
    return A


def diff_pauli_hamil_inner_product(circuit, index, qset_op_weight, param: Value, phase_shift):
    new_blocks = circuit.blocks
    n_qubit = circuit.n_qubit

    i_block, _ = circuit.get_block_index_by_param_index(index)

    block = new_blocks[i_block]

    assert isinstance(block, Rotation)

    diff_block = Controlled(GateGroup(PauliGate(block.qset, block.pauli_ops), GlobalPhase(-np.pi / 2)), (n_qubit,), [0])
    diff_block_pauli = Controlled(GateGroup(PauliGate(qset_op_weight[0], qset_op_weight[1])), (n_qubit,), [1])

    new_blocks.append(diff_block_pauli)
    new_blocks.insert(i_block + 1, diff_block)
    new_blocks.insert(0, GateGroup(SingleRotation(3, n_qubit, phase_shift)))
    new_blocks.insert(0, GateGroup(Hadamard(n_qubit)))

    new_circuit = MetaCircuit(circuit.n_qubit + 1, new_blocks)

    node = DeviceCircuitNode(new_circuit, QubitOperator(f"X{n_qubit}"),
                             name="DiffHamilInnerProd", param=param)

    expv = node()
    return expv * (0.5 * qset_op_weight[2] * block.weight)


def diff_pauli_hamil_inner_product_imag(circuit, index, qset_op_weight, param: Value):
    return diff_pauli_hamil_inner_product(circuit, index, qset_op_weight, param, -pi / 2)


def C_mat_imag(circuit: MetaCircuit, operator: QubitOperator, param: Value):
    C = []
    for i_param in range(circuit.n_param):
        diff_hamil_innerp = Value(0.0)
        for qset_op_weight in operator.qset_op_weight():
            pauli_innerp = diff_pauli_hamil_inner_product_imag(circuit, i_param, qset_op_weight, param)
            diff_hamil_innerp = diff_hamil_innerp + pauli_innerp
        C.append(diff_hamil_innerp)
    C = Value.array(C)
    return C


def diff_pauli_hamil_inner_product_real(circuit, index, qset_op_weight, param: Value):
    return diff_pauli_hamil_inner_product(circuit, index, qset_op_weight, param, 0.0)


def C_mat_real(circuit: MetaCircuit, operator: QubitOperator, param: Value):
    C = []
    for i_param in range(circuit.n_param):
        diff_hamil_innerp = Value(0.0)
        for qset_op_weight in operator.qset_op_weight():
            pauli_innerp = diff_pauli_hamil_inner_product_real(circuit, i_param, qset_op_weight, param)
            diff_hamil_innerp = diff_hamil_innerp + pauli_innerp
        C.append(diff_hamil_innerp)
    C = Value.array(C)
    return C
