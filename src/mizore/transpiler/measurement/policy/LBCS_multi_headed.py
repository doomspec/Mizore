import jax.nn

from mizore.operators import QubitOperator
import jax.numpy as jnp
from jax.nn import softplus

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


def average_var(heads, head_ratios, pword_tensor_no_zero, coeffs):
    coverage = get_head_coverage(heads, head_ratios, pword_tensor_no_zero)
    var = jnp.sum(1.0 / coverage * (coeffs ** 2))
    return var


def loss(params, pword_batch, pword_coeff_batch):
    heads = softplus(params["heads"])
    heads_denom = 1.0 / jnp.sum(heads, axis=2)
    heads = jnp.einsum("iq, iqp -> iqp", heads_denom, heads)
    ratios = softplus(params["head_ratios"])
    ratios = ratios / jnp.sum(ratios)
    # print(jnp.sum(ratios))
    # print(jnp.sum(heads, axis=2))
    return average_var(heads, ratios, pword_batch, pword_coeff_batch)


def generate_multi_headed_LBCS_policy(hamil: QubitOperator, n_qubit, n_head, max_steps=500000):
    pauli_tensor, coeffs = get_operator_tensor(hamil, n_qubit)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    n_pauliwords = len(coeffs)
    batch_size = 630
    rng_key = jax.random.PRNGKey(1)
    rng_key, head_key = jax.random.split(rng_key)
    rng_key, head_ratio_key = jax.random.split(rng_key)
    params = {
        "head_ratios": jax.random.uniform(key=head_key, shape=(n_head,), minval=5, maxval=10),
        "heads": jax.random.uniform(key=head_ratio_key, shape=(n_head, n_qubit, 3), minval=5, maxval=10)
    }
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, pword_batch, pword_coeff_batch):
        loss_value, grads = jax.value_and_grad(loss)(params, pword_batch, pword_coeff_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    n_epoch = 0
    batch_n = 0
    with tqdm(range(max_steps), ncols=100) as pbar:
        for i in pbar:
            pword_batch = pauli_tensor[batch_n:batch_n + batch_size]
            pword_coeff_batch = coeffs[batch_n:batch_n + batch_size]
            params, opt_state, loss_value = step(params, opt_state, pword_batch, pword_coeff_batch)
            batch_n += batch_size
            if batch_n >= n_pauliwords:
                batch_n = 0
                n_epoch += 1
                if n_epoch % 5 == 0:
                    rng_key, shuffle_key = jax.random.split(rng_key)
                    pauli_tensor = jax.random.permutation(shuffle_key, pauli_tensor)
                    coeffs = jax.random.permutation(shuffle_key, coeffs)
                    # print(pauli_tensor.shape)
                if n_epoch % 30 == 0:
                    # print(loss_value)
                    pbar.set_description('Loss: {:.6f}'.format(loss_value))


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


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil

    # jax.config.update('jax_platform_name', 'cuda')
    hamil, _ = get_test_hamil("mol", "LiH_12_BK").remove_constant()
    generate_multi_headed_LBCS_policy(hamil, 12, 300)
