from qulacs import QuantumState
from qulacs.state import inner_product


def finite_diff_inner_product(circuit, index1, index2, eps=1e-4):
    n_param = circuit.n_param
    param = [0.0] * n_param

    state0 = QuantumState(circuit.n_qubit)
    circuit.get_backend_circuit(param).update_quantum_state(state0)

    param[index1] += eps
    state1 = QuantumState(circuit.n_qubit)
    circuit.get_backend_circuit(param).update_quantum_state(state1)

    param[index1] -= eps
    param[index2] += eps
    state2 = QuantumState(circuit.n_qubit)
    circuit.get_backend_circuit(param).update_quantum_state(state2)

    finite_diff_innerp = (inner_product(state1, state2) - inner_product(state0, state2) -
                          inner_product(state1,state0) + 1) / (eps ** 2)
    return finite_diff_innerp.real

