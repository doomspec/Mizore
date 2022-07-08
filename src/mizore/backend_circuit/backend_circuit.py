from __future__ import annotations

from typing import List

from mizore.backend_circuit.backend_state import BackendState
from mizore.backend_circuit.backend_state import BackendOperator

from mizore.backend_circuit.gate import Gate
from qulacs import QuantumCircuit as qulacsQCircuit

from mizore.operators import QubitOperator


class BackendCircuit:
    def __init__(self, n_qubit, gate_list: List[Gate]):
        qulacs_circuit = qulacsQCircuit(n_qubit)
        for gate in gate_list:
            qulacs_circuit.add_gate(gate.qulacs_gate)
        self.qulacs_circuit = qulacs_circuit

    @property
    def n_qubit(self):
        return self.qulacs_circuit.get_qubit_count()

    def get_quantum_state(self, dm=False) -> BackendState:
        state = BackendState(self.n_qubit, dm=dm)
        self.update_quantum_state(state)
        return state

    def update_quantum_state(self, state: BackendState):
        self.qulacs_circuit.update_quantum_state(state.qulacs_state)

    def get_expv_from_op(self, op: QubitOperator, dm=False):
        backend_op = BackendOperator(op)
        return self.get_expv(backend_op)

    def get_expv(self, ob: BackendOperator, dm=False):
        state = BackendState(self.n_qubit, dm=dm)
        self.qulacs_circuit.update_quantum_state(state.qulacs_state)
        return ob.get_expectation_value(state)

    def get_many_expv(self, obs: List[BackendOperator], dm=False):
        state = BackendState(self.n_qubit, dm=dm)
        self.qulacs_circuit.update_quantum_state(state.qulacs_state)
        res = [ob.get_expectation_value(state) for ob in obs]
        del state
        return res

    def __str__(self):
        return str(self.qulacs_circuit)