from qulacs.gate import X as qulacs_X, Y as qulacs_Y, Z as qulacs_Z, H as qulacs_H
from qulacs.gate import DenseMatrix
from mizore.backend_circuit.gate import Gate
import numpy as np

class GlobalPhase(Gate):
    def __init__(self, angle):
        Gate.__init__(self, (0,))
        self._angle = angle
        self._matrix = np.array([[np.exp(1j*angle), 0], [0, np.exp(1j*angle)]])

    @property
    def angle(self):
        return self._angle

    @property
    def qulacs_gate(self):
        return DenseMatrix(0, self._matrix)

    def get_inverse(self):
        return GlobalPhase(-self._angle)

class X(Gate):

    def __init__(self, target):
        Gate.__init__(self, (target,))
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_X(self.target)

    def get_inverse(self):
        return X(self.target)


class Y(Gate):

    def __init__(self, target):
        Gate.__init__(self, (target,))
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_Y(self.target)


class Z(Gate):

    def __init__(self, target):
        Gate.__init__(self, (target,))
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_Z(self.target)


class Hadamard(Gate):

    def __init__(self, target):
        Gate.__init__(self, (target,))
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_H(self.target)



