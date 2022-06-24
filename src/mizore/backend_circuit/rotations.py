from math import pi
from typing import List

from qulacs.gate import PauliRotation as qulacs_PauliRotation
from qulacs.gate import RX, RY, RZ

from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.one_qubit_gates import H
from mizore.backend_circuit.two_qubit_gates import CNOT
from mizore.operators.qubit_operator import QubitOperator

_qulacs_gate_map = {
    1: RX,
    2: RY,
    3: RZ
}

pauli_index_char_map = {
    1: 'X',
    2: 'Y',
    3: 'Z'
}


class SingleRotation(Gate):

    def __init__(self, pauli: int, target, angle: float):
        Gate.__init__(self, (target,))
        self.pauli = pauli
        self.target = target
        self.angle = angle

    @property
    def qulacs_gate(self):
        return _qulacs_gate_map[self.pauli](self.target, self.angle)

    def __str__(self):
        return "R{} at {} angle: {}".format(pauli_index_char_map[self.pauli], self.target, self.angle)


class PauliRotation(Gate):
    """
    Implement exp(iPt)
    """

    def __init__(self, qset: List[int], pauli_ops: List[int], angle: float):
        Gate.__init__(self, qset)
        self.pauli_ops = pauli_ops
        self.angle = angle

    @property
    def qulacs_gate(self):
        return qulacs_PauliRotation(self.qset, self.pauli_ops, self.angle)

    def __str__(self):
        return "PauliRotation{} at {} angle: {}".format(self.pauli_ops, self.qset, self.angle)

    def simple_reduce(self) -> List[Gate]:
        res = []

        for target, pauli in zip(self.qset, self.pauli_ops):
            if pauli == 1:
                res.append(H(target))
            elif pauli == 2:
                res.append(SingleRotation(1, target, -pi / 2))

        for i in range(1, len(self.qset)):
            res.append(CNOT(self.qset[i - 1], self.qset[i]))

        res.append(SingleRotation(3, self.qset[-1], self.angle))

        for i in range(1, len(self.qset)):
            res.append(CNOT(self.qset[-i - 1], self.qset[-i]))

        for target, pauli in zip(self.qset, self.pauli_ops):
            if pauli == 1:
                res.append(H(target))
            elif pauli == 2:
                res.append(SingleRotation(1, target, pi / 2))

        return res


class SingleEvolve(Gate):

    def __init__(self, pauli_sum: QubitOperator, angle: float):
        self.qset = pauli_sum.calc_qset()
        if len(self.qset) != 1:
            raise Exception("the pauli_sum inputted into SingleEvolve can only cover one qubit")
        Gate.__init__(self, self.qset)
        self.pauli_sum = pauli_sum

    @property
    def qulacs_gate(self):
        return None


class DoubleEvolve(Gate):

    def __init__(self, pauli_sum: QubitOperator, angle: float):
        self.qset = pauli_sum.calc_qset()
        if len(self.qset) != 2:
            raise Exception("the pauli_sum inputted into DoubleEvolve can only cover two qubits")
        Gate.__init__(self, self.qset)
        self.pauli_sum = pauli_sum

    def get_qulacs_gates(self):
        pass
