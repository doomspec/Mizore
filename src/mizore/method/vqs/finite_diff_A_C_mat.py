from mizore import jax_array, np_array
import jax.numpy as jnp

from mizore.backend_circuit.backend_circuit import BackendCircuit
from mizore.backend_circuit.multi_qubit_gates import PauliGate
from mizore.meta_circuit.meta_circuit import MetaCircuit


def finite_diff_inner_product(circuit: MetaCircuit, param, index1, index2, eps=1e-4):
    state0 = circuit.get_backend_state(param)

    param[index1] += eps
    state1 = circuit.get_backend_state(param)

    param[index1] -= eps
    param[index2] += eps
    state2 = circuit.get_backend_state(param)
    param[index2] -= eps

    finite_diff_innerp = (state1.inner_product(state2) - state0.inner_product(state2) -
                          state1.inner_product(state0) + 1) / (eps ** 2)

    return finite_diff_innerp


def get_A_by_finite_diff(circuit, param_):
    param = np_array(param_)
    n_param = circuit.n_param
    A = [[None for _ in range(n_param)] for _ in range(n_param)]
    for i in range(n_param):
        A[i][i] = finite_diff_inner_product(circuit, param, i, i)  # Only work for exp(-i/2 \theta)
        for j in range(i + 1, n_param):
            A[i][j] = finite_diff_inner_product(circuit, param, i, j)
            A[j][i] = finite_diff_inner_product(circuit, param, i, j)  # .conjugate()
    return jax_array(A)


def finite_diff_pauli_hamil_inner_product(circuit: MetaCircuit, index, qset_op_weight, param, eps=1e-5):
    state0 = circuit.get_backend_state(param)

    param[index] += eps
    state1 = circuit.get_backend_state(param)

    param[index] -= eps

    gates = circuit.get_gates(param)
    gates.append(PauliGate(qset_op_weight[0], qset_op_weight[1]))
    backend_circuit = BackendCircuit(circuit.n_qubit, gates)
    state_ops = backend_circuit.get_quantum_state()
    finite_diff_innerp = (state1.inner_product(state_ops) - state0.inner_product(state_ops)) / eps

    return finite_diff_innerp


def get_C_by_finite_diff(circuit, operator, param_):
    param = np_array(param_)
    C = []
    for i_param in range(circuit.n_param):
        diff_hamil_innerp = 0.0
        for qset_op_weight in operator.qset_op_weight():
            pauli_innerp = finite_diff_pauli_hamil_inner_product(circuit, i_param, qset_op_weight, param) * \
                           qset_op_weight[2]
            diff_hamil_innerp += pauli_innerp
        C.append(diff_hamil_innerp)
    C = jnp.array(C)
    return C
