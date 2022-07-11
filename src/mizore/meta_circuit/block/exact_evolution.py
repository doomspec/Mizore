from typing import List

from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.matrix_gate import MatrixGate
from mizore.meta_circuit.block import Block
from mizore.operators import QubitOperator

from numpy.linalg import eigh
from numpy import dot, diag, exp, array

from mizore.operators.matrix_form import get_operator_matrix


class ExactEvolution(Block):
    """
    Block for implementing e^{iHt}
    """
    def __init__(self, hamil: QubitOperator, init_time=0.0, to_decompose=False):
        """
        :param hamil: The hamiltonian to evolve
        :param init_time: The fixed time shift of the block
        :param to_decompose: If to_decompose is True, this block might be transpiled into physical evolution blocks,
                                such as first order Trotter
        """
        super().__init__(1, fixed_param=[init_time])
        self._hamil = hamil
        self.qset = hamil.get_qset()

        # We will diagonalize the Hamiltonian H into H=P*D*P^dagger
        self.vec_D = None
        self.mat_P = None


        self.to_decompose = to_decompose

    def diagonalize(self):
        hamiltonian_mat = get_operator_matrix(self._hamil, n_qubits=len(self.qset))
        self.vec_D, self.mat_P = eigh(hamiltonian_mat)

    def get_evolution_operator(self, evolve_time):
        # This part may change to use Sparse Matrix
        time_evol_op = dot(dot(self.mat_P, diag(exp(1j * evolve_time * self.vec_D))),
                           self.mat_P.T.conj())
        return time_evol_op

    def get_gates(self, params) -> List[Gate]:
        if self.vec_D is None:
            self.diagonalize()
        time_evol_op = self.get_evolution_operator(params[0] + self._fixed_param[0])
        time_evol_gate = MatrixGate(self.qset, time_evol_op)
        return [time_evol_gate]
