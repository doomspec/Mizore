from mizore.operators import QubitOperator
from mizore.transpiler.measurement.policy.policy import UniversalPolicy, get_heads_tensor_from_pwords


def L1_policy_maker(hamil: QubitOperator, n_qubit):
    hamil, const = hamil.remove_constant()
    weight = hamil.get_l1_norm_omit_const()
    heads = get_heads_tensor_from_pwords([pword for pword in hamil.terms], n_qubit)
    probs = [abs(coeff) / weight for coeff in hamil.terms.values()]
    head_children = [[pword] for pword in hamil.terms]
    return UniversalPolicy(heads, probs, head_children, hamil, n_qubit)
