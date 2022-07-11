import math
from typing import List

from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.one_qubit_gates import GlobalPhase
from mizore.backend_circuit.rotations import PauliRotation
from mizore.meta_circuit.block import Block
from mizore.operators import QubitOperator


class Trotter(Block):
    def __init__(self, hamil, max_delta_t, init_time=0.0, method=None):
        super().__init__(n_param=1, fixed_param=[init_time])
        self.max_delta_t = max_delta_t
        self.hamil = hamil

    def get_gates(self, params) -> List[Gate]:
        evol_time = self.fixed_param[0] + params[0]
        n_step = math.ceil(evol_time / self.max_delta_t)
        delta_t = evol_time / n_step
        return get_trotter(self.hamil, delta_t, n_step)


def get_trotter(hamil: QubitOperator, delta_t, n_step, order=1) -> List[Gate]:
    assert order == 1
    if order == 1:
        return first_order_trotter(hamil, delta_t, n_step)


def first_order_trotter(hamil: QubitOperator, delta_t, n_step) -> List[Gate]:
    hamil_no_const, const = hamil.remove_constant()
    gates_one_step: List[Gate] = []
    for qset, op, weight in hamil_no_const.qset_op_weight():
        gates_one_step.append(PauliRotation(qset, op, weight * delta_t * 2))
    gate_list = []
    for i in range(n_step):
        gate_list.extend(gates_one_step)
    if const != 0.0:
        gate_list.append(GlobalPhase(const * delta_t * n_step))
    return gate_list
