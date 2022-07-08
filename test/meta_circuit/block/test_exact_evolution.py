from qulacs import QuantumState

from mizore.backend_circuit.backend_circuit import BackendCircuit
from mizore.backend_circuit.backend_state import BackendState
from mizore.backend_circuit.multi_qubit_gates import PauliGate
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator

from numpy.testing import assert_array_almost_equal


def test_compare_to_simple_rotation():
    n_qubit = 3
    qset = [0, 1, 2]
    pauli = [1, 3, 2]
    op = QubitOperator.from_qset_op(qset, pauli)
    time = 1.0
    for i in range(10):
        init_state = BackendState(n_qubit)
        init_state.set_Haar_random_state()
        circuit1 = MetaCircuit(n_qubit, blocks=[GateGroup(PauliGate([1], [1])), ExactEvolution(op, init_time=time)])
        state1 = init_state.copy()
        circuit1.get_backend_circuit().update_quantum_state(state1)
        state_vec = state1.get_vector()
        circuit0 = MetaCircuit(n_qubit,
                               blocks=[GateGroup(PauliGate([1], [1])),
                                       Rotation(qset, pauli, angle_shift=time * 2)])
        state0 = init_state.copy()
        circuit0.get_backend_circuit().update_quantum_state(state0)
        state_vec_expect = state0.get_vector()

        assert_array_almost_equal(state_vec, state_vec_expect)
