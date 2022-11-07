import jax.nn

from mizore.operators import QubitOperator
import jax.numpy as jnp
from jax.nn import softplus

from mizore.transpiler.measurement.policy.policy import UniversalPolicy
from mizore.transpiler.measurement.policy.utils_for_tensor import get_pword_tensor


def get_shadow_from_para(para):
    shadow = softplus(para)
    shadow_prob = 1.0 / jnp.sum(shadow, axis=1)
    shadow = jnp.einsum("q, qp -> qp", shadow_prob, shadow)
    return shadow


def get_opt_objective(children_tensor_with_offset, coeffs):
    def obj(para):
        shadow = get_shadow_from_para(para)
        overlap = jnp.einsum("qp, iqp -> iq", shadow, children_tensor_with_offset)
        overlap = jnp.prod(overlap, axis=1)
        return jnp.sum((coeffs ** 2) / overlap)

    return obj


def LBCS_policy_maker(hamil: QubitOperator, n_qubit, grad_cutoff=1e-2, max_steps=10000, lr=1e-3):
    heads = [jnp.ones((n_qubit, 3)) / 3]
    probs = [1.0]
    heads_children = [[pword for pword in hamil.terms]]
    children_tensor = jnp.array([get_pword_tensor(pword, n_qubit) for pword in heads_children[0]])
    children_tensor_offset = 1.0 - jnp.sum(children_tensor, axis=2)
    children_tensor_offset = jnp.repeat(jnp.expand_dims(children_tensor_offset, axis=2), 3, axis=2)
    children_tensor_with_offset = children_tensor + children_tensor_offset
    coeffs = jnp.array([coeff for coeff in hamil.terms.values()])
    obj = get_opt_objective(children_tensor_with_offset, coeffs)
    obj = jax.jit(jax.value_and_grad(obj))
    para = jnp.ones((n_qubit, 3)) / 3
    for step in range(max_steps):
        value, grad = obj(para)
        para -= grad * lr
        if jnp.linalg.norm(grad) < grad_cutoff:
            break
    return UniversalPolicy([get_shadow_from_para(para)], probs, heads_children, hamil, n_qubit)
