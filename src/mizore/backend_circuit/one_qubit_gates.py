from qulacs.gate import X as qulacs_X, Y as qulacs_Y, Z as qulacs_Z, H as qulacs_H

from mizore.backend_circuit.gate import Gate


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


class H(Gate):

    def __init__(self, target):
        Gate.__init__(self, (target,))
        self.target = target

    @property
    def qulacs_gate(self):
        return qulacs_H(self.target)



