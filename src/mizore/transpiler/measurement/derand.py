import time
from collections import Counter

import numpy as np

from mizore.operators import QubitOperator
from mizore.transpiler.measurement.abst_measure import MeasureImpl
from mizore.transpiler.measurement.derand_helper import DerandomizationMeasurementBuilder
from mizore.transpiler.measurement.vectorize_helper import get_prime_pword_tensor, measure_res_for_pwords

pauli_op_marks = [3 * 5, 2 * 5, 2 * 3]
pauli_op_marks_map = {
    "X": 3 * 5,
    "Y": 2 * 5,
    "Z": 2 * 3
}


def map_pwords_to_measure(pwords_to_measure):
    mapped = [[pauli_op_marks_map[p] for p in pword] for pword in pwords_to_measure]
    return np.array(mapped)


class DerandMeasureImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot):
        super().__init__(observable, n_shot)
        self.n_qubit = observable.get_qset()[-1] + 1
        self.builder = DerandomizationMeasurementBuilder(observable, self.n_qubit)
        pwords_to_measure = self.builder.build(self.n_shot)
        self.pwords_to_measure = map_pwords_to_measure(pwords_to_measure)

    def estimate_by_state(self, state):
        # Prepare the accumulators
        n_qubit = self.n_qubit
        pwords_to_measure = self.pwords_to_measure
        hamil_pword_cover_count = {}
        hamil_pword_res = {}

        children = list(self.ob.terms.keys())
        children_prime_repr = get_prime_pword_tensor(children, n_qubit)
        children_res = np.zeros((len(children),))
        children_cover_count = np.zeros((len(children),))

        pwords_tuples = [tuple(int(p) for p in pw) for pw in pwords_to_measure]
        pword_counts = Counter(pwords_tuples)

        res_list = []
        for pword_tuple, n_shot in pword_counts.items():
            res = state.sample_pauli_measure_by_pauliword_nums(pword_tuple, n_shot, seed=time.time_ns())
            res_list.append(np.array(res))
        measure_res_list = np.concatenate(res_list, axis=0)

        for res in measure_res_list:
            res_num_for_pwords = measure_res_for_pwords(children_prime_repr, res)
            children_res += res_num_for_pwords
            children_cover_count += np.abs(res_num_for_pwords)

        for i_children in range(len(children)):
            child_tuple = children[i_children]
            hamil_pword_res[child_tuple] = children_res[i_children]
            hamil_pword_cover_count[child_tuple] = children_cover_count[i_children]

        estimation = 0.0
        for pword, res in hamil_pword_res.items():
            cover_count = hamil_pword_cover_count[pword]
            if cover_count != 0:
                estimation += res / cover_count * self.ob.terms[pword]
            else:
                raise Exception(f"{pword} is not measured!")

        return estimation


if __name__ == '__main__':
    from chemistry.simple_mols import large_12_qubit_lih, large_14_qubit_h2o
    from mizore.operators.spectrum import get_ground_state

    n_shot = 1000
    hamil = large_14_qubit_h2o()
    n_qubit = 14
    impl = DerandMeasureImpl(hamil, n_shot)
    energy, state = get_ground_state(n_qubit, hamil)
    mean, var = impl.get_mean_and_variance_from_multi_exp(200, state, pbar=True)
    print(energy)
    print(mean, var * n_shot)
    print(np.sqrt(var))
