import scipy, numpy
from functools import reduce
import scipy.sparse

# Make global definitions.
identity_csc = scipy.sparse.identity(2, format='csc', dtype=complex)
pauli_x_csc = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
pauli_y_csc = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
pauli_z_csc = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
pauli_matrix_map = {
    'I': identity_csc,
    'X': pauli_x_csc,
    'Y': pauli_y_csc,
    'Z': pauli_z_csc
}


def wrapped_kronecker(operator_1, operator_2):
    """Return the Kronecker product of two sparse.csc_matrix operators."""
    """
    Zi-Jian Zhang's modification
    Swapped operator_1 and operator_2 to make the amplitudes of the 
    result wavefunction in friendly indices to the simulator backend
    """
    return scipy.sparse.kron(operator_2, operator_1, 'csc')


def kronecker_operators(*args):
    """Return the Kronecker product of multiple sparse.csc_matrix operators."""
    return reduce(wrapped_kronecker, *args)


def qubit_operator_sparse(qubit_operator, n_qubits=None):
    """Initialize a Scipy sparse matrix from a QubitOperator.

    Args:
        qubit_operator(QubitOperator): instance of the QubitOperator class.
        n_qubits (int): Number of qubits.

    Returns:
        The corresponding Scipy sparse matrix.
    """

    # if n_qubits is None:
    #    n_qubits = count_qubits(qubit_operator)
    # if n_qubits < count_qubits(qubit_operator):
    #    raise ValueError('Invalid number of qubits specified.')

    # Construct the Scipy sparse matrix.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]

    # Loop through the terms.
    for qubit_term in qubit_operator.terms:
        tensor_factor = 0
        coefficient = qubit_operator.terms[qubit_term]
        sparse_operators = [coefficient]
        for pauli_operator in qubit_term:

            # Grow space for missing identity operators.
            if pauli_operator[0] > tensor_factor:
                identity_qubits = pauli_operator[0] - tensor_factor
                identity = scipy.sparse.identity(2 ** identity_qubits,
                                                 dtype=complex,
                                                 format='csc')
                sparse_operators += [identity]

            # Add actual operator to the list.
            sparse_operators += [pauli_matrix_map[pauli_operator[1]]]
            tensor_factor = pauli_operator[0] + 1

        # Grow space at end of string unless operator acted on final qubit.
        if tensor_factor < n_qubits or not qubit_term:
            identity_qubits = n_qubits - tensor_factor
            identity = scipy.sparse.identity(2 ** identity_qubits,
                                             dtype=complex,
                                             format='csc')
            sparse_operators += [identity]

        # Extract triplets from sparse_term.
        sparse_matrix = kronecker_operators(sparse_operators)
        values_list.append(sparse_matrix.tocoo(copy=False).data)
        (column, row) = sparse_matrix.nonzero()
        column_list.append(column)
        row_list.append(row)

    # Create sparse operator.
    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix(
        (values_list, (row_list, column_list)),
        shape=(n_hilbert, n_hilbert)).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def get_operator_matrix(operator, n_qubits=None):
    return qubit_operator_sparse(operator, n_qubits).todense()
