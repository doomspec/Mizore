import numpy as np

pauli_op_to_prime = {
    "X": 2,
    "Y": 3,
    "Z": 5
}


def get_prime_pword_tensor(pwords: list, n_qubit):
    """
    Args:
        pwords: A list of PauliTuple
    Returns:
        The prime tensor of the input, with dimension (len(pwords), n_qubit).
            Each entry of the tensor represent a Pauli operator.
            The mapping is I->0, X->2, Y->3, Z->5.
    """
    pword_tensors = []
    for pword in pwords:
        prime_repr = [0] * n_qubit
        for i_qubit, op in pword:
            prime_repr[i_qubit] = pauli_op_to_prime[op]
        pword_tensors.append(prime_repr)
    return np.array(pword_tensors)


def get_qwc_array(prod):
    return 1 - np.sign(np.sum(np.abs(np.abs(np.abs(prod) - 15) - 15), axis=1))


def measure_res_for_pwords(pwords_in_prime, measured_res_in_coprime):
    """
    Args:
        pwords_in_prime: A tensor of pauliwords represented by prime numbers
        measured_res_in_coprime: A measurement result in coprime numbers

    Returns:

    """
    # In prod, only 30, -30, and 0 are valid data. We should eliminate other values.
    prod = pwords_in_prime * measured_res_in_coprime
    # In is_qwc, pword whose prob only contains 30, -30 and 0 will be set to 1.
    is_qwc = get_qwc_array(prod)
    minus_30_count = np.sum(1 - np.abs(np.sign(prod + 30)), axis=1)
    measure_res = is_qwc * np.power(-1, minus_30_count)
    return measure_res
