from typing import List

from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.one_qubit_gates import GlobalPhase
from mizore.backend_circuit.rotations import PauliRotation
from mizore.meta_circuit.block import Block
from mizore.meta_circuit.block.gates import Gates
from mizore.operators import QubitOperator


def Trotter(hamil: QubitOperator, delta_t, n_step, order=1) -> Block:
    assert order == 1
    if order == 1:
        return first_order_trotter(hamil, delta_t, n_step)


def first_order_trotter(hamil: QubitOperator, delta_t, n_step) -> Block:
    hamil_no_const, const = hamil.remove_constant()
    gates_one_step: List[Gate] = []
    for qset, op, weight in hamil_no_const.qset_op_weight():
        gates_one_step.append(PauliRotation(qset, op, weight * delta_t * 2))
    gate_list = []
    for i in range(n_step):
        gate_list.extend(gates_one_step)
    if const != 0.0:
        gate_list.append(GlobalPhase(const * delta_t * n_step))
    return Gates.from_gate_list(gate_list)
