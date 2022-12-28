import math
import time
from itertools import chain
from typing import List, Set
import numpy as np
import optax
from jax._src.nn.functions import softplus
from tqdm import tqdm

from mizore.operators import QubitOperator
from mizore.transpiler.measurement.grouping_utils.qwc import is_qwc, get_covering_pauliword, qwc_pword_multiply
from mizore.transpiler.measurement.policy.policy import UniversalPolicy, get_heads_tensor_from_pwords
from mizore.transpiler.measurement.policy.training.LBCS_jax import var_on_mixed_state
from mizore.transpiler.measurement.policy.utils_for_tensor import get_operator_tensor, get_no_zero_pauliwords


def OGM_policy_maker(hamil: QubitOperator, n_term_cutoff=-1, optimize=None):
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

    if optimize is None:
        return policy

    pauli_tensor, coeffs = get_operator_tensor(hamil, n_qubit)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)

    ratios = optimize_ratios(heads_tensor, initial_probs, pauli_tensor, coeffs, optimize)

    return UniversalPolicy(heads_tensor, np.array(ratios), heads_children, hamil, n_qubit)


import jax.numpy as jnp
import jax


def get_ratio_from_param(ratio_param):
    ratios = softplus(ratio_param * 20)
    return ratios / jnp.sum(ratios)


def loss(params, heads, pword_batch, pword_coeff_batch):
    ratios = get_ratio_from_param(params["head_ratios"])
    return var_on_mixed_state(heads, ratios, pword_batch, pword_coeff_batch)


def optimize_ratios(heads, head_ratios, pauli_tensor, coeffs, args):
    n_step = args.__dict__.get("n_step", 500000)
    n_step_to_stop = args.__dict__.get("n_step_to_stop", 200)
    batch_size_bound = args.__dict__.get("batch_size_bound", 400)
    n_head = len(head_ratios)
    n_pauliwords = len(coeffs)
    batch_size = n_pauliwords // math.ceil(n_pauliwords / batch_size_bound) + 1
    print("batch_size", batch_size)

    rng_key = jax.random.PRNGKey(123)

    params = {
        "head_ratios": head_ratios
    }

    optimizer = optax.adam(learning_rate=0.005)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, pword_batch, pword_coeff_batch):
        loss_value, grads = jax.value_and_grad(loss)(params, heads, pword_batch, pword_coeff_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    n_epoch = 0
    batch_n = 0
    loss_in_epoch = []
    min_var = math.inf
    stop_count = 0
    with tqdm(range(n_step), ncols=100) as pbar:
        for i in pbar:
            pword_batch = pauli_tensor[batch_n:batch_n + batch_size]
            pword_coeff_batch = coeffs[batch_n:batch_n + batch_size]
            params, opt_state, loss_value = step(params, opt_state, pword_batch, pword_coeff_batch)
            batch_n += batch_size
            loss_in_epoch.append(float(loss_value))
            if batch_n >= n_pauliwords:
                batch_n = 0
                n_epoch += 1
                if n_epoch % 1 == 0:
                    rng_key, shuffle_key = jax.random.split(rng_key)
                    pauli_tensor = jax.random.permutation(shuffle_key, pauli_tensor)
                    coeffs = jax.random.permutation(shuffle_key, coeffs)

                var = sum(loss_in_epoch)
                pbar.set_description('Loss: {:.6f}, Epoch: {}'.format(var, n_epoch))
                loss_in_epoch = []
                if min_var - var > var * 1e-3:
                    stop_count = 0
                    min_var = var
                else:
                    stop_count += 1
                    if stop_count == n_step_to_stop:
                        print("n_step_to_stop", n_step_to_stop, "reached")
                        break

    return get_ratio_from_param(params["head_ratios"])
