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


def measure_res_for_pwords(pword_in_prime_tensors, measured_res_in_coprime_num):
    # In prod, only 30, -30, and 0 are valid data. We should eliminate other values.
    prod = pword_in_prime_tensors * measured_res_in_coprime_num
    is_qwc = 1 - np.sign(np.sum(np.abs(np.abs(np.abs(prod) - 15) - 15), axis=1))
    minus_30_count = np.sum(1 - np.abs(np.sign(prod + 30)), axis=1)
    measure_res = is_qwc * np.power(-1, minus_30_count)
    return measure_res
