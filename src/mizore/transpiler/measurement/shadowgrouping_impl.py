import math
from typing import List

import jax

from mizore.operators import QubitOperator
from tqdm import tqdm

import numpy as np

np.random.seed(0)

pauli_int_map = {
    "I": 0,
    "X": 2,
    "Y": 3,
    "Z": 5
}

copauli_char_map = {
    3 * 5: "X",
    2 * 5: "Y",
    2 * 3: "Z"
}


def pword_to_array(pword, n_qubit):
    array = [0] * n_qubit
    for i, p in pword:
        array[i] = pauli_int_map[p]
    return np.array(array)


def ops_to_arrays(ops, n_qubit):
    weights = []
    pwords = []
    for pword, coeff in ops.terms.items():
        weights.append(coeff)
        pwords.append(pword_to_array(pword, n_qubit))
    return np.stack(pwords), np.array(weights)


class Derandomization_weight():
    def __init__(self, focus_on_greedy=True):
        self.greedy = focus_on_greedy
        return

    def get_weights(self, w, eps, N_hits):
        inconf = np.exp(-0.5 * eps * eps * N_hits / (w ** 2))
        inconf -= np.exp(-0.5 * eps * eps * (N_hits + 1) / (w ** 2))
        if self.greedy:
            inconf[N_hits == 0] -= 1
            inconf[N_hits == 0] *= -1
        return inconf

    def __call__(self):
        return self.get_weights


class Energy_estimation_inconfidence_weight():
    def __init__(self, alpha=1):
        self.alpha = alpha
        assert alpha >= 1, "alpha has to be chosen larger or equal 1, but was {}.".format(alpha)
        return

    def get_weights_for_testing(self, w, eps, N_hits):
        inconf = self.alpha * w ** 2
        condition = N_hits != 0
        inconf[condition] /= self.alpha * (N_hits[condition] + 1) * N_hits[condition]
        return inconf

    def get_weights(self, w, eps, N_hits):
        inconf = self.alpha * np.abs(w)
        condition = N_hits != 0
        N = np.sqrt(N_hits[condition])
        Nplus1 = np.sqrt(N_hits[condition] + 1)
        inconf[condition] /= self.alpha * np.sqrt(N * Nplus1) / (Nplus1 - N)
        return inconf

    def __call__(self):
        return self.get_weights


class ShadowGroupingMeasurement:
    def __init__(self, operator: QubitOperator, nqubit, use_weight=True):
        """
        Args:
            operator: terms in the Hamiltonian
            nqubit: num of qubits
            use_weight: whether we use the values of the coefficients in the Hamiltonian
        """
        self.ops = operator
        self.nqubit = nqubit
        self.pwords, self.coeffs = ops_to_arrays(operator, nqubit)
        self.coeffs = np.abs(self.coeffs)
        # self.weight_function = Derandomization_weight(focus_on_greedy=False)
        self.weight_function = Energy_estimation_inconfidence_weight()
        self.eps = 1e-4
        if use_weight:
            self.coeffs = self.coeffs / np.max(self.coeffs)  # [abs(t.amp) / max_weight for t in self.terms]
        else:
            self.coeffs = np.array([1 for _ in range(len(self.pwords))])

    def build(self, min_nshot_a_term, max_nshot=None):
        return self.build_vectorized(min_nshot_a_term, max_nshot=max_nshot)

    def build_vectorized(self, min_nshot_a_term, max_nshot=None):
        """
        Args:
            min_nshot_a_term: number of measurements
            nqubit: number of qubits

        Returns: an array of Pauli strings
        """
        max_nshot = max_nshot or min_nshot_a_term * len(self.pwords) // 5
        return self._derandomized_classical_shadow_vectorized(self.pwords, min_nshot_a_term, max_nshot)

    def get_cost_function(self):
        cost_config = DerandomizationCost(self.nqubit, self.coeffs, shift=0)

        def cost_fun(hit_counts, not_matched_counts):
            V = cost_config.eta / 2 * hit_counts
            if_nqubit_smaller_than_maches_needed = np.heaviside(cost_config.nqubit - not_matched_counts, 1)
            V += if_nqubit_smaller_than_maches_needed * (- np.log(1 - cost_config.nu / (3 ** not_matched_counts)))
            cost_val = np.sum(np.exp(-V / np.array(self.coeffs)))
            return cost_val

        return cost_fun

    def _derandomized_classical_shadow_vectorized(self, observables, min_nshot_a_term, max_nshot):
        """
        Forked from
        https://github.com/hsinyuan-huang/predicting-quantum-properties/blob/master/data_acquisition_shadow.py
        and updated
        """
        hit_counts = np.array([0] * len(observables))
        results = []
        cost_fun = self.get_cost_function()
        self.coeffs = np.array(self.coeffs)
        with tqdm(range(max_nshot), ncols=100) as pbar:
            for n_step in pbar:
                weights = self.weight_function.get_weights(self.coeffs, self.eps, hit_counts)
                order = np.argsort(weights)
                # Measurement Pauli string
                measurement = np.zeros(self.nqubit)
                for idx in reversed(order):
                    pword = self.pwords[idx]
                    prod = (measurement * pword)
                    hit = (prod[prod != 0] == 2 * 3 * 5).all()
                    if hit:
                        # print(pword)
                        non_id = pword != 0
                        # overwrite those qubits that fall in the support of o
                        measurement[non_id] = 2 * 3 * 5 / pword[non_id]
                        # break sequence is case all identities in setting are exhausted
                        if np.min(measurement) > 0:
                            break
                #
                prod_for_matching = (measurement * self.pwords)
                # print((prod_for_matching == 0).shape)
                is_hit = np.logical_or(prod_for_matching == 0, prod_for_matching == 2 * 3 * 5)
                is_hit = 1 * is_hit.all(axis=-1)
                results.append(measurement)
                hit_counts += is_hit

                if n_step < 4:
                    print("M", measurement)
                    # print(hit_counts)
                else:
                    if n_step % 10 == 0:
                        pass
                        # print(np.min(hit_counts))
                    # exit(0)
                least_satisfied = np.min(hit_counts - np.ceil(min_nshot_a_term * self.coeffs))
                pbar.set_description(str(least_satisfied))
                if least_satisfied >= 0:
                    break
                if least_satisfied >= -10:
                    idx = np.argmin(hit_counts)
                    # print(self.pwords[idx])
                    # print(self.coeffs[idx])
                    no_id = self.pwords[idx] != 0
                    measure = np.zeros(self.nqubit)
                    measure[no_id] = 30 // self.pwords[idx][no_id]
                    # print(measure)
                    results.append(measure)
                    hit_counts[idx] += 1

        return results


class DerandomizationCost:
    def __init__(self, nqubit, weights, shift):
        self.weights = weights
        self.nqubit = nqubit
        self.shift = shift
        self.sum_log_value = 0
        self.sum_cnt = 0
        self.eta = 0.9
        self.nu = 1 - math.exp(-self.eta / 2)
