from typing import List

from mizore.backend_circuit.gate import Gate
from qulacs.gate import DenseMatrix


class MatrixGate(Gate):
    def __init__(self, qset: List[int], matrix):
        Gate.__init__(self, qset)
        self.matrix = matrix

    @property
    def qulacs_gate(self):
        return DenseMatrix(self.qset, self.matrix)

    def __str__(self):
        return "MatrixGate at {}".format(self.qset)

    def simple_reduce(self) -> List[Gate]:
        raise Exception("This gate is non-physical")
