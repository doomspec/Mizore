from itertools import chain
from typing import Dict, Tuple, Set, List, Iterable

from mizore.operators.qubit_operator import PauliTuple, QubitOperator


def is_qwc(term1: PauliTuple, term2: PauliTuple):
    """
    Returns:
        Whether two pauliwords are qubit-wise commute. Pauliwords should be PauliTuples
    """
    # TODO: It is good if someone can verify this
    j = 1
    for i in range(len(term1)):
        index1 = term1[i][0]
        index2 = term2[j - 1][0]
        while index2 < index1 and j < len(term2):
            index2 = term2[j][0]
            j += 1
        if index1 == index2:
            if term1[i][1] != term2[j - 1][1]:
                return False
    return True


def qwc_pword_multiply(pword1, pword2):
    n_qubit = max(pword1[-1][0], pword2[-1][0]) + 1
    aux_list = [0] * n_qubit
    pauli_list = [0] * n_qubit
    for pauli in chain(pword1, pword2):
        aux_list[pauli[0]] += 1
        pauli_list[pauli[0]] = pauli[1]
    prod = []
    for i in range(len(aux_list)):
        if aux_list[i] == 1:
            prod.append((i, pauli_list[i]))
    return tuple(prod)


def get_qwc_graph(hamil: QubitOperator) -> Dict[PauliTuple, Set]:
    """
    Construct the graph which links every pair of pauliwords that commute

    Returns:
        A mapping that maps pauliwords to its adjacent pauliwords in the graph.
        Pauliwords are represented by PauliTuple
    """
    terms = list(hamil.terms.keys())
    graph: Dict[Tuple, Set] = {term: set() for term in terms}
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if is_qwc(terms[i], terms[j]):
                graph[terms[i]].add(terms[j])
                graph[terms[j]].add(terms[i])
    return graph


def get_covering_pauliword(qwc_group: Iterable[PauliTuple]) -> PauliTuple:
    qset = set()
    pauli_map = {}
    for pauliword in qwc_group:
        for pauli in pauliword:
            if pauli[0] not in qset:
                qset.add(pauli[0])
                pauli_map[pauli[0]] = pauli[1]
    qset = list(qset)
    qset.sort()
    return tuple(((i, pauli_map[i]) for i in qset))


def get_prob_from_groupings(qwc_group: List[Set], hamil: QubitOperator):
    """
    Calculate the sum of (absolute coefficients in the clique) divided by (L1-norm of the Hamiltonian)
    Args:
        qwc_group: A list of cliques
        hamil: The hamiltonian
    Returns:
        A list of shot ratios for each clique (or say group).
    """
    assert () not in hamil.terms.keys()
    weight_sum = hamil.get_l1_norm_omit_const()
    group_shot_ratios = []
    term_added = set()
    for clique in qwc_group:
        clique_weight_sum = 0.0
        for term in clique:
            if term not in term_added:
                term_added.add(term)
            else:
                assert False
            clique_weight_sum += abs(hamil.terms[term])
        group_shot_ratios.append(clique_weight_sum / weight_sum)
    return group_shot_ratios
