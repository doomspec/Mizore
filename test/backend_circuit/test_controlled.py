from cmath import exp
from math import pi, sqrt

from mizore.backend_circuit.one_qubit_gates import GlobalPhase, Hadamard
from mizore.meta_circuit.block.controlled import Controlled
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from numpy.testing import assert_array_almost_equal


def test_controlled_global_phase():
    for i in [-0.2, 0.4, -0.8, 1.0]:
        blocks = [Gates(Hadamard(1)), Controlled(Gates(GlobalPhase(-pi / 2 * i)), (1,))]
        circuit = MetaCircuit(2, blocks=blocks)
        state = circuit.get_backend_state()
        assert_array_almost_equal(state.get_vector(), [1 / sqrt(2), 0.0, 1 / sqrt(2) * exp(-pi / 2 * i * 1j), 0.0])
