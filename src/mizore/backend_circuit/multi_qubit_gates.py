from typing import List

from mizore.backend_circuit.gate import Gate
from qulacs.gate import Pauli as qulacs_Pauli
from .one_qubit_gates import X, Y, Z

qulacs_pauli_gate_map = [None, X, Y, Z]


class PauliGate(Gate):
    """
    Implement Pauli Gate
    """

    def __init__(self, qset: List[int], pauli_ops: List[int]):
        Gate.__init__(self, qset)
        self.pauli_ops = pauli_ops

    @property
    def qulacs_gate(self):
        return qulacs_Pauli(self.qset, self.pauli_ops)

    def __str__(self):
        return "Pauli{} at {}".format(self.pauli_ops, self.qset)

    def simple_reduce(self) -> List[Gate]:
        return [qulacs_pauli_gate_map[pauli](target) for target, pauli in zip(self.qset, self.pauli_ops)]
