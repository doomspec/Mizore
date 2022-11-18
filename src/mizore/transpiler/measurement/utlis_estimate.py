from collections import Counter
import time
from mizore.transpiler.measurement.vectorize_helper import get_prime_pword_tensor, measure_res_for_pwords, get_qwc_array
import numpy as np

pauli_op_marks_map = {
    "X": 3 * 5,
    "Y": 2 * 5,
    "Z": 2 * 3
}


def map_pwords_to_measure(pwords_to_measure):
    mapped = [[pauli_op_marks_map[p] for p in pword] for pword in pwords_to_measure]
    return np.array(mapped)


def estimate_by_list_of_pwords(ob, pwords_in_tensor, res_generator):
    # Prepare the accumulators
    n_qubit = ob.n_qubit
    hamil_pword_cover_count = {}
    hamil_pword_res = {}

    children = list(ob.terms.keys())
    children_prime_repr = get_prime_pword_tensor(children, n_qubit)
    children_res = np.zeros((len(children),))
    children_cover_count = np.zeros((len(children),))

    pwords_tuples = [tuple(int(p) for p in pw) for pw in pwords_in_tensor]
    pword_counts = Counter(pwords_tuples)

    res_list = []
    for coprime_pword, n_shot in pword_counts.items():
        res = res_generator(coprime_pword, n_shot, seed=time.time_ns())
        # res = state.sample_pauli_measure_by_coprime_pword()
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
            estimation += res / cover_count * ob.terms[pword]
        else:
            raise Exception(f"{pword} is not measured!")

    return estimation


def estimate_by_list_of_pwords_on_state(ob, pwords_in_tensor, state):
    def res_generator(coprime_pword, n_shot, seed=0):
        return state.sample_pauli_measure_by_coprime_pword(coprime_pword, n_shot, seed)

    return estimate_by_list_of_pwords(ob, pwords_in_tensor, res_generator)


def get_totally_random_measure_res(coprime_pword, n_shot):
    random_sign = np.random.binomial(1, 0.5, (n_shot, len(coprime_pword)))
    random_sign = (-1) ** random_sign
    res = coprime_pword * random_sign
    return res


def estimate_by_list_of_pwords_on_totally_random_state(ob, pwords_in_tensor):
    def res_generator(coprime_pword, n_shot, seed=0):
        return get_totally_random_measure_res(coprime_pword, n_shot)

    return estimate_by_list_of_pwords(ob, pwords_in_tensor, res_generator)


def average_var_by_list_of_pwords(ob, pwords_in_tensor):
    n_qubit = ob.n_qubit
    children = list(ob.terms.keys())
    children_prime_repr = get_prime_pword_tensor(children, n_qubit)
    children_cover_count = np.zeros((len(children),))

    pwords_tuples = [tuple(int(p) for p in pw) for pw in pwords_in_tensor]
    pword_counts = Counter(pwords_tuples)

    for coprime_pword, n_shot in pword_counts.items():
        prod = coprime_pword * children_prime_repr
        is_qwc = get_qwc_array(prod)
        children_cover_count += is_qwc * n_shot

    var = 0.0
    for i in range(len(children)):
        var += (ob.terms[children[i]] ** 2) * (1 / children_cover_count[i])

    return var


if __name__ == '__main__':
    get_totally_random_measure_res(np.array([3, 2, 1, 3, 3]), 10, 3)
