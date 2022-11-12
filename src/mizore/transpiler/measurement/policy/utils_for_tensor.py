import jax.numpy as jnp

from mizore.operators import QubitOperator

pauli_map = {"X": 0, "Y": 1, "Z": 2}


def get_pword_tensor(pword, n_qubit):
    pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit][pauli_map[pauli]] = 1.0
    return jnp.array(pauli_tensor, copy=False)


def get_operator_tensor(op: QubitOperator, n_qubit):
    coeffs = []
    pwords = []
    for pword, coeff in op.terms.items():
        pwords.append(get_pword_tensor(pword, n_qubit))
        coeffs.append(coeff)
    return jnp.array(pwords), jnp.array(coeffs)


def get_no_zero_pauliwords(pword_tensor):
    anti_qubit_mask = 1.0 - jnp.sum(pword_tensor, axis=-1)
    anti_qubit_mask = jnp.expand_dims(anti_qubit_mask, axis=2)
    anti_qubit_mask = anti_qubit_mask.repeat(3, axis=2)
    no_zero_pauliwords = pword_tensor + anti_qubit_mask
    return no_zero_pauliwords


if __name__ == '__main__':
    test_op = QubitOperator("X1 X2 X3") + QubitOperator("Z0 Z2")
    p, c = get_operator_tensor(test_op, 4)
    print(p)
    print(get_no_zero_pauliwords(p))
