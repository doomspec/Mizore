import math
import time
from itertools import chain
from typing import List, Set
import numpy as np
from jax._src.nn.functions import softplus

from mizore.operators import QubitOperator
from mizore.transpiler.measurement.grouping_utils.qwc import is_qwc, get_covering_pauliword, qwc_pword_multiply
from mizore.transpiler.measurement.policy.policy import UniversalPolicy, get_heads_tensor_from_pwords


def OGM_policy_maker(hamil: QubitOperator, n_term_cutoff=-1, optimize=False):
    n_qubit = hamil.n_qubit
    hamil, constant = hamil.remove_constant()
    assert constant == 0.0

    start_time = time.time()

    covering_pwords = []
    hamil_terms = {k: abs(v) for k, v in hamil.terms.items()}
    ranked_pwords = sorted(hamil_terms, key=hamil_terms.get, reverse=True)
    abs_coeffs = [hamil_terms[pword] for pword in ranked_pwords]
    # Do cutoff
    if n_term_cutoff != -1:
        ranked_pwords = ranked_pwords[:n_term_cutoff]
    added_terms = set()
    groups = []
    group_mapping = [set() for pword in ranked_pwords]
    initial_probs = []
    for i in range(len(ranked_pwords)):
        if i in added_terms:
            continue
        covering_pword = ranked_pwords[i]
        group_mapping[i].add(len(groups))
        group = {ranked_pwords[i]}
        initial_prob = 0.0
        for j in chain(range(i + 1, len(ranked_pwords)), range(i)):
            if is_qwc(covering_pword, ranked_pwords[j]):
                covering_pword = get_covering_pauliword([covering_pword, ranked_pwords[j]])
                added_terms.add(j)
                group.add(ranked_pwords[j])
                initial_prob += abs_coeffs[j]
                group_mapping[j].add(len(groups))
        covering_pwords.append(covering_pword)
        groups.append(group)
        initial_probs.append(initial_prob)
    initial_probs = np.array(initial_probs)
    initial_probs = initial_probs / np.sum(initial_probs)

    # return GeneralGroupingInfo(group_mapping, ranked_pwords, covering_pwords, initial_probs, abs_coeffs)
    heads_tensor = get_heads_tensor_from_pwords(covering_pwords, n_qubit)
    heads_children = [list(group) for group in groups]

    policy = UniversalPolicy(heads_tensor, initial_probs, heads_children, hamil, n_qubit)

    time_used = time.time() - start_time

    if not optimize:
        return policy

    from mizore.transpiler.measurement.policy.training.LBCS_jax import get_operator_tensor, get_no_zero_pauliwords, \
        var_on_mixed_state
    import jax.numpy as jnp
    import jax
    from tqdm import trange

    pauli_tensor, coeffs = get_operator_tensor(hamil, n_qubit)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)

    # print(heads_tensor)
    def var_by_ratio(param_ratios, pword_batch, pword_coeff_batch):
        ratios = softplus(param_ratios * 10) / 10
        ratios = ratios / jnp.sum(ratios)
        return var_on_mixed_state(heads_tensor, ratios, pword_batch, pword_coeff_batch)

    def loss(param_ratios):
        return var_by_ratio(param_ratios, pauli_tensor, coeffs) * (1 - 1 / (2 ** n_qubit + 1))

    obj = jax.jit(jax.value_and_grad(loss))

    lr = 1e-3
    grad_cutoff = 1e-2
    param = initial_probs * 10

    init_var, _ = obj(param)
    print("init_var", init_var)
    start_time = time.time()
    min_var = math.inf
    stop_count = 0
    for step in trange(100000):
        value, grad = obj(param)
        param -= grad * lr
        if value < min_var:
            min_var = value
            stop_count = 0
        else:
            stop_count += 1
            if stop_count > 300:
                break

    ratios = softplus(param * 10) / 10
    ratios = ratios / jnp.sum(ratios)

    return UniversalPolicy(heads_tensor, np.array(ratios), heads_children, hamil, n_qubit)
