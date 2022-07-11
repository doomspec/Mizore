from typing import Union, List

import numpy

from mizore.backend_circuit.matrix_gate import MatrixGate
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.operators.matrix_form import get_operator_matrix
from scipy.linalg import eigh
import numpy as np

def krylov_kernel_mat_classical(ref_circuit: MetaCircuit, hamil: QubitOperator,
                                kernel_op: Union[QubitOperator, None], n_basis: int, delta: float):
    assert n_basis > 1
    M: List[List[Union[complex, None]]] = [[None for _ in range(n_basis)] for _ in range(n_basis)]
    kernel_op_gate = None
    if kernel_op is not None:
        kernel_op_mat = get_operator_matrix(kernel_op, ref_circuit.n_qubit)
        kernel_op_gate = MatrixGate(list(range(ref_circuit.n_qubit)), kernel_op_mat)
    for i in range(n_basis):
        circuit_i = ref_circuit.replica()
        if i != 0:
            circuit_i.add_blocks([ExactEvolution(hamil, init_time=delta * i)])
        if kernel_op_gate is not None:
            circuit_i.add_blocks([Gates(kernel_op_gate)])
        psi_i_H = circuit_i.get_backend_state()
        circuit_j = ref_circuit.replica()
        circuit_j.add_blocks([ExactEvolution(hamil, init_time=delta * i)])
        psi_j = circuit_j.get_backend_state()
        M[i][i] = psi_i_H.inner_product(psi_j)
        for j in range(i + 1, n_basis):
            psi_j = circuit_j.get_backend_state([delta * (j - i)])
            M[i][j] = psi_i_H.inner_product(psi_j)
            M[j][i] = M[i][j].conjugate()
    return np.array(M)



def H_mat_classical(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float):
    return krylov_kernel_mat_classical(ref_circuit, hamil, hamil, n_basis, delta)


def S_mat_classical(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float):
    return krylov_kernel_mat_classical(ref_circuit, hamil, None, n_basis, delta)


def quantum_krylov_single_ref_classical(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float):
    H = H_mat_classical(ref_circuit, hamil, n_basis, delta)
    S = S_mat_classical(ref_circuit, hamil, n_basis, delta)
    S += np.eye(len(S))*1e-12
    try:
        #eigv = eigh(np.linalg.inv(S)@H)[0]
        eigv = eigh(H, S, eigvals_only=True)
    except:
        print("Eigenvalue of S", eigh(S, eigvals_only=True))
    return eigv
