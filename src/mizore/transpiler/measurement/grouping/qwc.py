from typing import Dict, Tuple, Set, List

from mizore.operators import QubitOperator


def is_qwc(term1, term2):
    j = 1
    for i in range(len(term1)):
        index1 = term1[i][0]
        index2 = term2[0][0]
        while index2 < index1 and j < len(term2):
            index2 = term2[j][0]
            j += 1
        if index1 == index2:
            if term1[i][1] != term2[j - 1][1]:
                return False
    return True


def get_qwc_complement_graph(hamil: QubitOperator):
    terms = list(hamil.terms.keys())
    graph: Dict[Tuple, Set] = {term: set() for term in terms}
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if not is_qwc(terms[i], terms[j]):
                graph[terms[i]].add(terms[j])
                graph[terms[j]].add(terms[i])
    return graph


def available_color(adjacent_colors_sorted):
    for i in range(len(adjacent_colors_sorted) - 1):
        if adjacent_colors_sorted[i] + 1 != adjacent_colors_sorted[i + 1]:
            return adjacent_colors_sorted[i] + 1
    return adjacent_colors_sorted[-1] + 1


def largest_degree_first_coloring(c_graph: Dict[Tuple, Set]):
    degree_map = {term: len(value) for term, value in c_graph.items()}
    terms_ranked = sorted(degree_map, key=degree_map.get, reverse=True)
    color_map = {term: -1 for term in c_graph.keys()}
    highest_color = -1
    for term in terms_ranked:
        adjacent_colors = [color_map[adjacent] for adjacent in c_graph[term]]
        if len(adjacent_colors) == 0:
            color = 0
        else:
            adjacent_colors.sort()
            color = available_color(adjacent_colors)
        if color > highest_color:
            highest_color = color
        color_map[term] = color
    cliques = [set() for _ in range(highest_color + 1)]
    for term, value in color_map.items():
        cliques[value].add(term)
    return cliques


def get_qwc_cliques_by_LDF(hamil: QubitOperator):
    cg = get_qwc_complement_graph(hamil)
    cliques = largest_degree_first_coloring(cg)
    return cliques


def get_shot_num_map_from_cliques(cliques: List[Set], hamil: QubitOperator):
    assert () not in hamil.terms.keys()
    weight_sum = sum(list(hamil.terms.values()))
    clique_shot_ratio = []
    clique_pauliwords = []
    for clique in cliques:
        clique_weight_sum = 0.0
        clique_pauli_map = {}
        for term in clique:
            clique_weight_sum += abs(hamil.terms[term])
            for pauli_tuple in term:
                clique_pauli_map[pauli_tuple[0]] = pauli_tuple[1]
        clique_shot_ratio.append(clique_weight_sum/weight_sum)
        qset = list(clique_pauli_map.keys())
        qset.sort()
        clique_pauliwords.append(tuple((i, clique_pauli_map[i]) for i in qset))
    return clique_pauliwords, clique_shot_ratio
