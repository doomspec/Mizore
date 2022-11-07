from itertools import chain
from typing import Dict, Tuple, Set, List, Iterable

from mizore.operators import QubitOperator
from mizore.operators.qubit_operator import PauliTuple
from mizore.transpiler.measurement.policy.policy import get_heads_tensor_from_pwords, UniversalPolicy
from mizore.transpiler.measurement.policy.utils_for_grouping import is_qwc, get_covering_pauliword, \
    get_prob_from_groupings


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


def LDF_policy_maker(hamil: QubitOperator, n_qubit):
    cliques = get_qwc_cliques_by_LDF(hamil)
    probs = get_prob_from_groupings(cliques, hamil)
    heads = [get_covering_pauliword(clique) for clique in cliques]
    heads = get_heads_tensor_from_pwords(heads, n_qubit)
    head_children = [list(clique) for clique in cliques]
    return UniversalPolicy(heads, probs, head_children, hamil, n_qubit)
