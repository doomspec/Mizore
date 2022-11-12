from mizore.backend_circuit.backend_state import BackendState
from mizore.operators import QubitOperator
from mizore.operators.matrix_form import qubit_operator_sparse, get_operator_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numpy


def get_first_k_eigenstates(k, n_qubits, operator: QubitOperator, sparse=True, initial_guess=None):
    """
    Adopted from openfermion.linalg.get_ground_state
    """
    if sparse:
        sparse_operator = qubit_operator_sparse(operator, n_qubits)

        values, vectors = eigsh(sparse_operator,
                                k=k + 1,
                                v0=initial_guess,
                                which='SA',
                                maxiter=1e7)
    else:
        mat_operator = get_operator_matrix(operator, n_qubits)
        values, vectors = eigh(mat_operator)

    order = numpy.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eigenvalue = values[:k]
    eigenstate = vectors[:, :k]
    eigenstate = eigenstate.T

    return eigenvalue, eigenstate


def get_ground_state(n_qubits, operator: QubitOperator, sparse=True, initial_guess=None):
    energy, vec = get_first_k_eigenstates(1, n_qubits, operator, sparse=sparse, initial_guess=initial_guess)
    state = BackendState(n_qubits)
    state.set_vector(vec[0])
    return energy[0], state
