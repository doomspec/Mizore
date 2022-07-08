from __future__ import annotations

from typing import Union

from mizore.operators import QubitOperator

from qulacs import Observable


class BackendOperator:
    def __init__(self, op: QubitOperator):
        self.op = op
        self.n_qubit = -1
        self.qulacs_op: Union[None, Observable] = None

    def init(self, n_qubit):
        self.qulacs_op = make_qulacs_operator(n_qubit, self.op)
        self.n_qubit = n_qubit

    def get_expectation_value(self, state):
        if state.n_qubit != self.n_qubit:
            self.init(state.n_qubit)
        return self.qulacs_op.get_expectation_value(state.qulacs_state)


def make_qulacs_operator(n_qubit, op: QubitOperator) -> Observable:
    operator = Observable(n_qubit)
    for qset, pauli_ops, weight in op.qset_op_weight():
        operator.add_operator(weight, pauliword_to_qulacs_string(qset, pauli_ops))
    return operator


index_to_pauli_name = ["I", "X", "Y", "Z"]


def pauliword_to_qulacs_string(qset, pauli_ops):
    return " ".join(map(lambda index, pauli: "{} {}".format(index_to_pauli_name[pauli], str(index)), qset, pauli_ops))
