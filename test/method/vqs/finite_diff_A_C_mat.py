import qulacs
from qulacs import QuantumState
from qulacs.state import inner_product
from mizore import jax_array
from mizore.meta_circuit.block.rotation import Rotation
import jax.numpy as jnp


def finite_diff_inner_product(circuit, index1, index2, eps=1e-5):
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
                          inner_product(state1, state0) + 1) / (eps ** 2)
    # print(finite_diff_innerp)
    return finite_diff_innerp


def get_A_by_finite_diff(circuit):
    n_param = circuit.n_param
    A = [[None for _ in range(n_param)] for _ in range(n_param)]
    for i in range(n_param):
        A[i][i] = finite_diff_inner_product(circuit, i, i)  # $0.25  # Only work for exp(-i/2 \theta)
        for j in range(i + 1, n_param):
            A[i][j] = finite_diff_inner_product(circuit, i, j)
            A[j][i] = finite_diff_inner_product(circuit, i, j)  # .conjugate()
    return jax_array(A)


def finite_diff_pauli_hamil_inner_product(circuit, index, qset_op_weight, eps=1e-5):
    n_param = circuit.n_param

    param = [0.0] * n_param
    state0 = QuantumState(circuit.n_qubit)
    circuit.get_backend_circuit(param).update_quantum_state(state0)

    param[index] += eps
    state1 = QuantumState(circuit.n_qubit)
    circuit.get_backend_circuit(param).update_quantum_state(state1)

    param = [0.0] * n_param
    state_ops = QuantumState(circuit.n_qubit)
    backend_circuit: qulacs.QuantumCircuit = circuit.get_backend_circuit(param)
    backend_circuit.add_multi_Pauli_gate(qset_op_weight[0], qset_op_weight[1])
    backend_circuit.update_quantum_state(state_ops)
    finite_diff_innerp = (inner_product(state1, state_ops) - inner_product(state0, state_ops)) / eps

    return finite_diff_innerp


def get_C_by_finite_diff(circuit, operator):
    C = []
    for i_param in range(circuit.n_param):
        diff_hamil_innerp = 0.0
        for qset_op_weight in operator.qset_ops_weight():
            pauli_innerp = finite_diff_pauli_hamil_inner_product(circuit, i_param, qset_op_weight) * qset_op_weight[2]
            diff_hamil_innerp += pauli_innerp
        C.append(diff_hamil_innerp)
    C = jnp.array(C)
    return C
