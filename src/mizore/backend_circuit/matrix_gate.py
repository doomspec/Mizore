from typing import List

from mizore.backend_circuit.gate import Gate
from qulacs.gate import DenseMatrix


class MatrixGate(Gate):
    def __init__(self, qset: List[int], matrix, annotate=None):
        Gate.__init__(self, qset)
        self.matrix = matrix
        self.annotate = annotate

    @property
    def qulacs_gate(self):
        return DenseMatrix(self.qset, self.matrix)

    def simple_reduce(self) -> List[Gate]:
        raise Exception("This gate is non-physical")

    def __str__(self):
        if self.annotate is not None:
            return f"MatrixGate at {self.qset} ({self.annotate})"
        else:
            return Gate.__str__(self)
