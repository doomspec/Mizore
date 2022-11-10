import math, random
from mizore.operators import QubitOperator


class Term:
    def __init__(self, amp, p_string):
        self.amp = amp
        self.p_string = p_string

    def __repr__(self) -> str:
        return "Term({}, {})".format(self.amp, self.p_string)


def to_p_string(tuples, nqubit):
    dict = {}
    for t in tuples:
        dict[t[0]] = t[1]
    results = []
    for j in range(nqubit):
        if j not in dict:
            results.append("I")
        else:
            results.append(dict[j])
    return "".join(results)


def to_terms(operator: QubitOperator, nqubit) -> [Term]:
    results = []
    for k, v in operator.terms.items():
        results.append(Term(v, to_p_string(k, nqubit)))
    return results


class DerandomizationMeasurementBuilder:
    def __init__(self, operator: QubitOperator, nqubit, use_weight=True):
        """
        Args:
            operator: terms in the Hamiltonian
            nqubit: num of qubits
            use_weight: whether we use the values of the coefficients in the Hamiltonian
        """
        self.terms = to_terms(operator, nqubit)
        self.nqubit = nqubit
        max_weight = 0
        for t in self.terms:
            a = abs(t.amp)
            if a > max_weight:
                max_weight = a
        if use_weight:
            self.weights = [abs(t.amp) / max_weight for t in self.terms]
        else:
            self.weights = [1 for _ in range(len(self.terms))]

    def build(self, nshot):
        """
        Args:
            nshot: number of measurements
            nqubit: number of qubits

        Returns: an array of Pauli strings
        """
        result = []
        for t in self.terms:
            paulis = []
            for i, c in enumerate(t.p_string):
                if c == "I":
                    continue
                paulis.append((c, i))
            result.append(paulis)
        return self._derandomized_classical_shadow(result, nshot)

    @classmethod
    def _matches(cls, qubit_i, pauli_candidate, observable):
        for pauli, pos in observable:
            if pos != qubit_i:
                continue
            if pauli != pauli_candidate:
                return -1
            else:
                return 1
        return 0

    def _derandomized_classical_shadow(self, observables, nshot):
        """
        Forked from
        https://github.com/hsinyuan-huang/predicting-quantum-properties/blob/master/data_acquisition_shadow.py
        and updated
        """
        hit_counts = [0] * len(observables)
        results = []
        cost = DerandomizationCost(self.nqubit, self.weights, shift=0)
        for _ in range(nshot):
            # A single round of parallel measurement over "system_size" number of qubits
            not_matched_counts = [len(P) for P in observables]
            # Measurement Pauli string
            measurement = []

            for qubit_index in range(self.nqubit):
                cost_of_outcomes = dict([("X", 0), ("Y", 0), ("Z", 0)])

                for pauli_candidate in ["X", "Y", "Z"]:
                    # Assume the dice rollout to be "dice_roll_pauli"
                    candidate_matched_counts = not_matched_counts.copy()
                    for i, observable in enumerate(observables):
                        match = self._matches(qubit_index, pauli_candidate, observable)
                        if match == -1:
                            candidate_matched_counts[i] += 100 * (self.nqubit + 10)  # impossible to measure
                        if match == 1:
                            candidate_matched_counts[i] -= 1
                    cost_of_outcomes[pauli_candidate] = cost.value(hit_counts,
                                                                   candidate_matched_counts)
                ps = ["X", "Y", "Z"]
                random.shuffle(ps)
                for pauli_candidate in ps:
                    if min(cost_of_outcomes.values()) < cost_of_outcomes[pauli_candidate]:
                        continue
                    measurement.append(pauli_candidate)
                    for i, observable in enumerate(observables):
                        match = self._matches(qubit_index, pauli_candidate, observable)
                        if match == -1:
                            not_matched_counts[i] += 100 * (self.nqubit + 10)  # impossible to measure
                        if match == 1:
                            not_matched_counts[i] -= 1  # match up one Pauli X/Y/Z
                    break

            results.append(measurement)

            for i, observable in enumerate(observables):
                if not_matched_counts[i] == 0:  # finished measuring all qubits
                    hit_counts[i] += 1
        return results


class DerandomizationCost:
    def __init__(self, nqubit, weights, shift):
        self.weights = weights
        self.nqubit = nqubit
        self.shift = shift
        self.sum_log_value = 0
        self.sum_cnt = 0
        self.eta = 0.9

    def value(self, hit_counts, not_matched_counts):
        nu = 1 - math.exp(-self.eta / 2)
        shift = 0
        cost = 0
        for i, zipitem in enumerate(zip(hit_counts, not_matched_counts)):
            hit_count, matches_needed = zipitem
            if self.nqubit < matches_needed:
                V = self.eta / 2 * hit_count
            else:
                V = self.eta / 2 * hit_count - math.log(1 - nu / (3 ** matches_needed))
            cost += math.exp(-V / self.weights[i] - shift)
            self.sum_log_value += V / self.weights[i]
            self.sum_cnt += 1
        return cost
