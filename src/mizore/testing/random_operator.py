from numpy.random import default_rng
from mizore.operators import QubitOperator

pauli_name_map = ["I", "X", "Y", "Z"]


def to_term_key(op_with_I):
    term_key = []
    for j in range(len(op_with_I)):
        if op_with_I[j] != 0:
            term_key.append((j, pauli_name_map[op_with_I[j]]))
    return tuple(term_key)


def sample_single_term(n_qubit, seed):
    sampled_pauli = default_rng(seed).integers(0, 3, (n_qubit,))
    sampled_weight = default_rng(seed + 1).uniform(-1.0, 1.0)
    return to_term_key(sampled_pauli), sampled_weight


def get_random_operator(n_qubit, n_term, weight_sum, seed):
    if n_term > 4 ** n_qubit / 2:
        raise Exception("Too many terms to sample. The efficiency cannot be ensured.")
    sampled_pauli = default_rng(seed).integers(0, 3, (n_term, n_qubit))
    sampled_weight = default_rng(seed + 1).uniform(-1.0, 1.0, (n_term,))

    terms = {}
    for i in range(n_term):
        op_with_I = sampled_pauli[i]
        terms[to_term_key(op_with_I)] = sampled_weight[i]

    if () in terms:
        del terms[()]

    sampled_weight_additional = []
    seed_i = 9  # This number is ad hoc
    while len(terms) != n_term:
        seed_i += 1
        term_key, weight = sample_single_term(n_qubit, seed=seed + seed_i)
        if len(term_key) == 0:
            continue
        if term_key in terms:
            continue
        sampled_weight_additional.append(weight)
        terms[term_key] = weight

    sampled_weight_norm = 0.0
    for weight in terms.values():
        sampled_weight_norm += abs(weight)

    op = QubitOperator()
    op.terms = terms
    op = op * (weight_sum / sampled_weight_norm)
    return op


if __name__ == '__main__':
    hamil = get_random_operator(8, 100, 100, 100)
    print(hamil)
