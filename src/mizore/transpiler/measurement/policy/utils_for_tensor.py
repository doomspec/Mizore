import jax.numpy as jnp

pauli_map = {"X": 0, "Y": 1, "Z": 2}


def get_pword_tensor(pword, n_qubit):
    pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit][pauli_map[pauli]] = 1.0
    return jnp.array(pauli_tensor, copy=False)
