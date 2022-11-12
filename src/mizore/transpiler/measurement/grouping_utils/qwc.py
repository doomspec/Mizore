from itertools import chain
from typing import Dict, Tuple, Set, List, Iterable

from mizore.operators import QubitOperator
from mizore.operators.qubit_operator import PauliTuple


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


def get_qwc_complement_graph(hamil: QubitOperator) -> Dict[PauliTuple, Set]:
    """
    Construct the complement graph which lines every pauliword that doesn't commute

    Returns:
        A mapping that maps pauliwords to its adjacent pauliwords in the graph.
        Pauliwords are represented by PauliTuple
    """
    terms = list(hamil.terms.keys())
    graph: Dict[Tuple, Set] = {term: set() for term in terms}
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if not is_qwc(terms[i], terms[j]):
                graph[terms[i]].add(terms[j])
                graph[terms[j]].add(terms[i])
    return graph


def lowest_available_color(adjacent_colors_sorted):
    for i in range(len(adjacent_colors_sorted) - 1):
        if adjacent_colors_sorted[i] == adjacent_colors_sorted[i + 1]:
            continue
        if adjacent_colors_sorted[i] + 1 != adjacent_colors_sorted[i + 1]:
            return adjacent_colors_sorted[i] + 1
    return adjacent_colors_sorted[-1] + 1


def largest_degree_first_coloring(c_graph: Dict[Tuple, Set]) -> List[Set[PauliTuple]]:
    """
    Args:
        c_graph: The complement graph
    Returns:
        The cliques in the complement graph.
        The cliques are represented by a list that maps the index of color to a set of PauliTuples
    """
    degree_map = {term: len(edge_list) for term, edge_list in c_graph.items()}
    terms_ranked = sorted(degree_map, key=degree_map.get, reverse=True)
    color_map = {term: -1 for term in c_graph.keys()}
    highest_color = -1
    for term in terms_ranked:
        adjacent_colors = [color_map[adjacent] for adjacent in c_graph[term]]
        if len(adjacent_colors) == 0:
            color = 0
        else:
            adjacent_colors.sort()
            color = lowest_available_color(adjacent_colors)
        if color > highest_color:
            highest_color = color
        color_map[term] = color
    cliques = [set() for _ in range(highest_color + 1)]
    for term, value in color_map.items():
        cliques[value].add(term)
    return cliques


def get_qwc_cliques_by_LDF(hamil: QubitOperator) -> List[Set[PauliTuple]]:
    cg = get_qwc_complement_graph(hamil)
    cliques = largest_degree_first_coloring(cg)
    return cliques


def get_prob_from_groupings(cliques: List[Set], hamil: QubitOperator):
    """
    Calculate the sum of (absolute coefficients in the clique) divided by (L1-norm of the Hamiltonian)
    Args:
        cliques: A list of cliques
        hamil: The hamiltonian
    Returns:
        A list of shot ratios for each clique (or say group).
    """
    assert () not in hamil.terms.keys()
    weight_sum = hamil.get_l1_norm_omit_const()
    group_shot_ratios = []
    for clique in cliques:
        clique_weight_sum = 0.0
        for term in clique:
            clique_weight_sum += abs(hamil.terms[term])
        group_shot_ratios.append(clique_weight_sum / weight_sum)
    return group_shot_ratios


def get_covering_pauliword(clique: Iterable[PauliTuple]) -> PauliTuple:
    qset = set()
    pauli_map = {}
    for pauliword in clique:
        for pauli in pauliword:
            if pauli[0] not in qset:
                qset.add(pauli[0])
                pauli_map[pauli[0]] = pauli[1]
    qset = list(qset)
    qset.sort()
    return tuple(((i, pauli_map[i]) for i in qset))


if __name__ == '__main__':
    res = qwc_pword_multiply(((0, "X"), (1, "Z")), ((0, "X"), (1, "Z"), (2, "Y"), (3, "Z")))
    print(res)
