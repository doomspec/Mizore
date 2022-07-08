from qulacs import QuantumState

from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.fixed_block import FixedBlock
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.real_time_evol import real_evol_gradient
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner

from numpy.testing import assert_array_almost_equal


def test_single_qubit():
    n_qubit = 1
    hamil = QubitOperator("Z0") + QubitOperator("X0") + QubitOperator("Y0")

    blocks = [Rotation((0,), (1,), angle_shift=1.0),
              Rotation((0,), (3,), angle_shift=1.5),
              Rotation((0,), (1,), angle_shift=1.0)]

    circuit = MetaCircuit(n_qubit, blocks)

    blocks2 = [FixedBlock(block) for block in blocks] + [ExactEvolution(hamil)]

    circuit_ref = MetaCircuit(n_qubit, blocks2)

    step_size = 1e-3

    for i in range(2):
        curr_time = 0.0
        param = Value([0.0]*circuit.n_param)

        init_state = QuantumState(n_qubit)
        init_state.set_Haar_random_state()

        evol_grad, A, C = real_evol_gradient(circuit, hamil, param)

        cg = CompGraph([evol_grad])
        CircuitRunner() | cg

        param = param + evol_grad*step_size
        curr_time += step_size

        state = init_state.copy()
        circuit.get_backend_circuit(param.value()).update_quantum_state(state)

        state_ref = init_state.copy()
        circuit_ref.get_backend_circuit([curr_time]).update_quantum_state(state_ref)

        assert_array_almost_equal(state_ref.get_vector(), state.get_vector())
