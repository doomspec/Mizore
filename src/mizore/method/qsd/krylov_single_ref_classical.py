from typing import Union, List

import numpy

from mizore.backend_circuit.backend_op import BackendOperator
from mizore.backend_circuit.matrix_gate import MatrixGate
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.operators.matrix_form import get_operator_matrix
from scipy.linalg import eigh
import numpy as np


def krylov_kernel_mat_classical(ref_circuit: MetaCircuit, hamil: QubitOperator,
                                use_hamil_kernel: bool, n_basis: int, delta: float):
    assert n_basis > 1
    M: List[List[Union[complex, None]]] = [[None for _ in range(n_basis)] for _ in range(n_basis)]
    kernel_op_gate = None
    ref_circuit = ref_circuit.replica()
    if use_hamil_kernel:
        kernel_op_mat = get_operator_matrix(hamil, ref_circuit.n_qubit)
        kernel_op_gate = MatrixGate(list(range(ref_circuit.n_qubit)), kernel_op_mat)

    self_innerp_list = []
    innerp_list = [None]

    backend_hamil = BackendOperator(hamil)
    circuit_0 = ref_circuit.replica()
    if kernel_op_gate is not None:
        # Eval <psi|H|psi>
        self_innerp_list.append(circuit_0.get_expectation_value(backend_hamil))
        circuit_0.add_blocks([Gates(kernel_op_gate)])
    else:
        self_innerp_list.append(1.0)

    psi_0_H = circuit_0.get_backend_state()
    circuit_i = MetaCircuit(ref_circuit.n_qubit, gates=ref_circuit.get_gates())
    circuit_i.add_blocks([ExactEvolution(hamil, init_time=0.0)])

    for i in range(1, n_basis):
        psi_i = circuit_i.get_backend_state([i * delta])
        innerp_list.append(psi_0_H.inner_product(psi_i))
        if kernel_op_gate is not None:
            self_innerp_list.append(psi_i.get_expv(backend_hamil))
        else:
            self_innerp_list.append(1.0)

    for i in range(n_basis):
        M[i][i] = self_innerp_list[i]
        for j in range(i + 1, n_basis):
            M[i][j] = innerp_list[j - i]
            M[j][i] = M[i][j].conjugate()
    return np.array(M)


def H_mat_classical(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float):
    return krylov_kernel_mat_classical(ref_circuit, hamil, True, n_basis, delta)


def S_mat_classical(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float):
    return krylov_kernel_mat_classical(ref_circuit, hamil, False, n_basis, delta)


def quantum_krylov_single_ref_classical(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float,
                                        get_H_S=False):
    hamil_no_const, const = hamil.remove_constant()
    H = H_mat_classical(ref_circuit, hamil_no_const, n_basis, delta)
    S = S_mat_classical(ref_circuit, hamil_no_const, n_basis, delta)
    S += np.eye(len(S)) * 1e-12
    try:
        # eigv = eigh(np.linalg.inv(S)@H)[0]
        eigv = eigh(H, S, eigvals_only=True)
    except:
        print("Eigenvalue of S", eigh(S, eigvals_only=True))
    if not get_H_S:
        return eigv + const
    else:
        return eigv + const, H, S
