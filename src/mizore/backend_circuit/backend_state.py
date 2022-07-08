from __future__ import annotations

from typing import Union, List

from qulacs import DensityMatrix, QuantumState
from qulacs.state import inner_product

from mizore.backend_circuit.backend_op import BackendOperator


class BackendState:
    def __init__(self, n_qubit, dm=False):
        self.n_qubit = n_qubit
        self.qulacs_state: Union[QuantumState, DensityMatrix]
        if n_qubit == -1:
            return
        if not dm:
            self.qulacs_state = QuantumState(n_qubit)
        else:
            self.qulacs_state = DensityMatrix(n_qubit)

    def inner_product(self, other: BackendState):
        return inner_product(self.qulacs_state, other.qulacs_state)

    def get_expv(self, ob: BackendOperator):
        return ob.get_expectation_value(self)

    def get_many_expv(self, obs: List[BackendOperator]):
        return [ob.get_expectation_value(self) for ob in obs]

    def set_Haar_random_state(self):
        self.qulacs_state.set_Haar_random_state()

    def copy(self):
        copied = BackendState(n_qubit=-1)
        copied.n_qubit = self.n_qubit
        copied.qulacs_state = self.qulacs_state.copy()
        return copied

    def get_matrix(self):
        return self.qulacs_state.get_matrix()

    def get_vector(self):
        return self.qulacs_state.get_vector()
