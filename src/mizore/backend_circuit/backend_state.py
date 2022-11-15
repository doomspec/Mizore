from __future__ import annotations

from typing import Union, List

from qulacs import DensityMatrix, QuantumState
from qulacs.state import inner_product
from qulacs.gate import H, RX

from mizore.backend_circuit.backend_op import BackendOperator
from mizore.operators.qubit_operator import PauliTuple

import time
from math import log2, pi


class BackendState:
    def __init__(self, n_qubit, dm=False):
        self.n_qubit = n_qubit
        self.qulacs_state: Union[QuantumState, DensityMatrix]
        if n_qubit == -1:
            return
        if not dm:
            self.qulacs_state = QuantumState(n_qubit)
        else:
            self.qulacs_state = DensityMatrix(n_qubit)

    def inner_product(self, other: BackendState):
        return inner_product(self.qulacs_state, other.qulacs_state)

    def get_expv(self, ob: BackendOperator):
        return ob.get_expectation_value(self)

    def get_many_expv(self, obs: List[BackendOperator]):
        return [ob.get_expectation_value(self) for ob in obs]

    def set_Haar_random_state(self):
        self.qulacs_state.set_Haar_random_state()

    def copy(self):
        copied = BackendState(n_qubit=-1)
        copied.n_qubit = self.n_qubit
        copied.qulacs_state = self.qulacs_state.copy()
        return copied

    def get_matrix(self):
        return self.qulacs_state.get_matrix()

    def get_vector(self):
        return self.qulacs_state.get_vector()

    def get_zero_probability(self, qset):
        query_list = [2] * self.n_qubit
        for index in qset:
            query_list[index] = 0
        return self.qulacs_state.get_marginal_probability(query_list)

    def set_vector(self, vector):
        self.qulacs_state.load(vector)

    def sample_1_qset(self, count, seed) -> List:
        """
        Args:
            count: Number of samples required
            seed: Seed for random sampling
        Returns: Simulate the measurement of the state and return the indices of the qubits whose outcome is 1.
        """
        sampled = self.qulacs_state.sampling(count, seed)
        sampled_qset = [number_to_qset(number) for number in sampled]
        return sampled_qset

    seed_shift = 0

    def sample_pauli_measure(self, qset, op, count, seed=None, qset_only=True):
        if count == 0:
            return []
        state = self.copy()
        # Change basis
        for i in range(len(qset)):
            if op[i] == 1:
                H(qset[i]).update_quantum_state(state.qulacs_state)
            if op[i] == 2:
                RX(qset[i], -pi / 2).update_quantum_state(state.qulacs_state)
        if seed is None:
            seed = int(time.time()) * 7
        sampled_num_for_qsets = state.qulacs_state.sampling(count, int((seed + self.seed_shift * 11) % 10000000))
        self.seed_shift += 1
        """
        # Reverse operation
        # I disabled this because it seems the performance is similar
        # We just need to copy the state every time
        for i in range(len(qset)):
            if op[i] == 1:
                H(qset[i]).update_quantum_state(state.qulacs_state)
            if op[i] == 2:
                RX(qset[i], pi / 2).update_quantum_state(state.qulacs_state)
        """

        if qset_only:
            sampled_qsets = [number_to_qset_with_superset(number, qset) for number in sampled_num_for_qsets]
        else:
            sampled_qsets = [number_to_qset(number) for number in sampled_num_for_qsets]
        return sampled_qsets

    def sample_pauli_measure_by_pauliword_tuple(self, pauliword: PauliTuple, count, seed=None, qset_only=True):
        if count == 0:
            return []
        state = self.copy()
        # Change basis
        for pauli in pauliword:
            if pauli[1] == 'X':
                H(pauli[0]).update_quantum_state(state.qulacs_state)
            elif pauli[1] == 'Y':
                RX(pauli[0], -pi / 2).update_quantum_state(state.qulacs_state)
        if seed is None:
            seed = int(time.time())
        sampled_num_for_qsets = state.qulacs_state.sampling(count, int((seed + self.seed_shift * 13) % 10000000))
        self.seed_shift += 1
        if qset_only:
            sampled_qsets = [number_to_qset_with_superset2(number, pauliword) for number in sampled_num_for_qsets]
        else:
            sampled_qsets = [number_to_qset(number) for number in sampled_num_for_qsets]
        return sampled_qsets

    def sample_pauli_measure_by_coprime_pword(self, pword, n_shot, seed=None):
        """
        Args:
            pword: The pauliword to measure, in the form of iterable coprime numbers
            n_shot: Number of shots
            seed: seed for RNG

        Returns: A list of lists. The content of sublists has the same absolute value as pword.
                    The result is represented by the signs.

        """
        state = self.copy()
        # Change basis
        for i in range(len(pword)):
            pauli = pword[i]
            if pauli == 3 * 5:
                H(i).update_quantum_state(state.qulacs_state)
            elif pauli == 2 * 5:
                RX(i, -pi / 2).update_quantum_state(state.qulacs_state)
        if seed is None:
            seed = int(time.time())
        sampled_num_for_qsets = state.qulacs_state.sampling(n_shot, int((seed + self.seed_shift * 13) % 10000000))
        self.seed_shift += 1
        return [number_to_res_list_in_flipped_signs(number, pword) for number in sampled_num_for_qsets]


def number_to_res_list_in_flipped_signs(number, pword):
    res = list(pword)
    if number == 0:
        return res
    highest = int(log2(number)) + 1
    for i in range(0, highest):
        if int((number % (1 << (i + 1))) / (1 << i)) == 1:
            res[i] *= -1
    return res


def number_to_qset(number):
    if number == 0:
        return ()
    highest = int(log2(number)) + 1
    qset = []
    for i in range(0, highest):
        if int((number % (1 << (i + 1))) / (1 << i)) == 1:
            qset.append(i)
    return qset


def number_to_qset_with_superset(number, super_qset):
    if number == 0:
        return ()
    qset = []
    for i in super_qset:
        if int((number % (1 << (i + 1))) / (1 << i)) == 1:
            qset.append(i)
    return qset


def number_to_qset_with_superset2(number, super_pauliword: PauliTuple):
    if number == 0:
        return ()
    qset = []
    for pauli in super_pauliword:
        i = pauli[0]
        if int((number % (1 << (i + 1))) / (1 << i)) == 1:
            qset.append(i)
    return qset
