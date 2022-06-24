from mizore.backend_circuit.gate import Gate
from qulacs.gate import DepolarizingNoise as qulacs_Depolar


class NoiseGate(Gate):
    def __init__(self, qset, prob):
        super().__init__(qset)
        self.is_noise = True
        self.prob = prob

    def __str__(self):
        return "{} at {} prob: {}".format(self.__class__.__name__, self.qset, self.prob)


class Depolarizing(NoiseGate):
    def __init__(self, target, prob):
        NoiseGate.__init__(self, (target,), prob)

    @property
    def qulacs_gate(self):
        return qulacs_Depolar(self.qset[0], self.prob)