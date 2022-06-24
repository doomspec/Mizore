from qulacs.gate import CNOT as qulacs_CNOT
from qulacs.gate import CP as qulacs_CP
from qulacs.gate import CZ as qulacs_CZ
from qulacs.gate import SWAP as qulacs_SWAP

from mizore.backend_circuit.gate import Gate


class CNOT(Gate):

    def __init__(self, control, target):
        Gate.__init__(self, (control, target))
        self.control = control
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_CNOT(self.control, self.target)


class CZ(Gate):

    def __init__(self, control, target):
        Gate.__init__(self, (control, target))
        self.control = control
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_CZ(self.control, self.target)


class CP(Gate):
    def __init__(self, control, target):
        Gate.__init__(self, (control, target))
        self.control = control
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_CP(self.control, self.target)


class SWAP(Gate):
    def __init__(self, target1, target2):
        Gate.__init__(self, (target1, target2))
        self.target1 = target1
        self.target2 = target2

    @property
    def qulacs_gate(self):
        return qulacs_SWAP(self.target1, self.target2)
