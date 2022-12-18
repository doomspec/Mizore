import jax.nn

from mizore.operators import QubitOperator
import jax.numpy as jnp
from jax.nn import softplus

from mizore.transpiler.measurement.policy.OGM import OGM_policy_maker
from mizore.transpiler.measurement.policy.policy import UniversalPolicy
from mizore.transpiler.measurement.policy.utils_for_tensor import get_pword_tensor, get_operator_tensor, \
    get_no_zero_pauliwords
import numpy as np
from jax.nn import softplus
import optax
from tqdm import tqdm


def get_head_coverage(heads, head_ratios, pword_tensor_no_zero):
    shadow_coverage = jnp.einsum("nqp, sqp -> nsq", pword_tensor_no_zero, heads)
    coverage = jnp.prod(shadow_coverage, axis=-1)
    coverage = jnp.einsum("s, ns -> n", head_ratios, coverage)
    return coverage


def var_on_mixed_state(heads, head_ratios, pword_tensor_no_zero, coeffs):
    coverage = get_head_coverage(heads, head_ratios, pword_tensor_no_zero)
    var = jnp.sum(1.0 / coverage * (coeffs ** 2))
    return var


@jax.jit
def loss(params, pword_batch, pword_coeff_batch):
    heads = softplus(params["heads"] * 10) * 10
    heads_denom = 1.0 / jnp.sum(heads, axis=2)
    heads = jnp.einsum("iq, iqp -> iqp", heads_denom, heads)
    ratios = softplus(params["head_ratios"])
    ratios = ratios / jnp.sum(ratios)
    # print(jnp.sum(ratios))
    # print(jnp.sum(heads, axis=2))
    return var_on_mixed_state(heads, ratios, pword_batch, pword_coeff_batch)


def bilevel_grad_modifier(grads, n_epoch):
    grads["head_ratios"] *= 0.01


def generate_multi_headed_LBCS_policy(hamil: QubitOperator, n_qubit, n_head,
                                      init_params=None, batch_size=-1, grad_modifier=bilevel_grad_modifier,
                                      max_steps=500000, seed=1234):
    pauli_tensor, coeffs = get_operator_tensor(hamil, n_qubit)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    n_pauliwords = len(coeffs)
    if batch_size == -1:
        batch_size = n_pauliwords / 3 + 3
    rng_key = jax.random.PRNGKey(seed)
    rng_key, head_key = jax.random.split(rng_key)
    rng_key, head_ratio_key = jax.random.split(rng_key)
    if init_params is None:
        params = {
            "head_ratios": jax.random.uniform(key=head_key, shape=(n_head,), minval=5, maxval=10),
            "heads": jax.random.uniform(key=head_ratio_key, shape=(n_head, n_qubit, 3), minval=5, maxval=10)
        }
    else:
        params = init_params

    optimizer = optax.adam(learning_rate=0.005)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, pword_batch, pword_coeff_batch, n_epoch):
        loss_value, grads = jax.value_and_grad(loss)(params, pword_batch, pword_coeff_batch)
        grad_modifier(grads, n_epoch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    n_epoch = 0
    batch_n = 0
    with tqdm(range(max_steps), ncols=100) as pbar:
        for i in pbar:
            pword_batch = pauli_tensor[batch_n:batch_n + batch_size]
            pword_coeff_batch = coeffs[batch_n:batch_n + batch_size]
            params, opt_state, loss_value = step(params, opt_state, pword_batch, pword_coeff_batch, n_epoch)
            batch_n += batch_size
            if batch_n >= n_pauliwords:
                batch_n = 0
                n_epoch += 1
                if n_epoch % 5 == 0:
                    rng_key, shuffle_key = jax.random.split(rng_key)
                    pauli_tensor = jax.random.permutation(shuffle_key, pauli_tensor)
                    coeffs = jax.random.permutation(shuffle_key, coeffs)
                if n_epoch % 30 == 0:
                    var = loss(params, pauli_tensor, coeffs)
                    pbar.set_description('Loss: {:.6f}, Epoch: {}'.format(var, n_epoch))


def multi_headed_LBCS_from_file(hamil, n_qubit, path):
    with open(path, "rb") as f:
        shadow = np.load(f)
        ratio = np.load(f)
    # ratio = jnp.array(ratio)
    ratio = ratio / np.sum(ratio)
    print(np.sum(ratio))
    # shadow = jnp.array(shadow)
    n_head = len(ratio)
    heads_children = [[pword for pword in hamil.terms] for h in range(n_head)]
    return UniversalPolicy(shadow, ratio, heads_children, hamil, n_qubit)


def get_one_by_one_grad_modifier(n_head, n_qubit):
    head_mask = jnp.zeros((n_head, n_qubit, 3))
    head_ratio_mask = jnp.zeros((n_head,))

    def one_by_one_grad_modifier(grads, n_epoch):
        offset = 0  # 1 - (n_epoch % 20000) // 10000
        idx = (n_epoch // 2) % n_head
        grads["head_ratios"] *= 1.0
        grads["head_ratios"] *= head_ratio_mask.at[idx].set(1.0) + offset
        grads["heads"] *= head_mask.at[idx, :].set(1.0) + offset

    return one_by_one_grad_modifier


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil

    # jax.config.update('jax_platform_name', 'cuda')
    n_head = 300
    hamil, _ = get_test_hamil("mol", "LiH_12_BK").remove_constant()
    n_qubit = hamil.n_qubit
    ogm = OGM_policy_maker(hamil, hamil.n_qubit, len(hamil.terms))
    init_n_head = len(ogm.heads_tensor)
    print(f"OGM produces {init_n_head} heads!")
    rng_key = jax.random.PRNGKey(123)
    if init_n_head <= n_head:
        init_params = {
            "head_ratios": jnp.concatenate(
                [10 * ogm.heads_ratio, jax.random.uniform(rng_key, (n_head - init_n_head,))]),
            "heads": jnp.concatenate(
                [ogm.heads_tensor,
                 jax.random.uniform(rng_key, (n_head - init_n_head, n_qubit, 3), minval=5, maxval=10)]),
        }
    else:
        print(f"Use only {n_head} heads")
        init_params = {
            "head_ratios": 10 * ogm.heads_ratio[:n_head],
            "heads": 10 * ogm.heads_tensor[:n_head],
        }
    generate_multi_headed_LBCS_policy(hamil, hamil.n_qubit, n_head, batch_size=320, init_params=init_params, )

if __name__ == '__main__1':
    from mizore.testing.hamil import get_test_hamil

    # jax.config.update('jax_platform_name', 'cuda')
    n_head = 300
    hamil, _ = get_test_hamil("mol", "LiH_12_BK").remove_constant()
    n_qubit = hamil.n_qubit
    ogm = OGM_policy_maker(hamil, hamil.n_qubit, len(hamil.terms))
    init_n_head = len(ogm.heads_tensor)
    assert init_n_head < n_head
    rng_key = jax.random.PRNGKey(123)
    # init_method = "ones"
    init_method = "random"
    if init_method == "ones":
        init_params = {
            "head_ratios": jnp.ones((n_head,)),
            "heads": jnp.ones((n_head, n_qubit, 3)),
        }
    elif init_method == "random":
        init_params = {
            "head_ratios": jax.random.uniform(rng_key, (n_head,)),
            "heads": jax.random.uniform(rng_key, (n_head, n_qubit, 3), minval=5, maxval=10),
        }
    else:
        assert False
    generate_multi_headed_LBCS_policy(hamil, hamil.n_qubit, n_head, batch_size=300, init_params=init_params,
                                      grad_modifier=get_one_by_one_grad_modifier(n_head, n_qubit))
