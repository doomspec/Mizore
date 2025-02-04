from itertools import chain
from typing import List, Set
import numpy as np
from mizore.operators import QubitOperator
from mizore.transpiler.measurement.grouping_utils.qwc import is_qwc, get_covering_pauliword, qwc_pword_multiply


class GeneralGroupingInfo:
    def __init__(self, group_mapping, ranked_pwords, covering_pwords, probs, abs_coeffs):
        self.group_mapping = group_mapping
        self.ranked_pwords = ranked_pwords
        self.covering_pwords = covering_pwords
        self.probs = np.array(probs)
        self.abs_coeffs = np.array(abs_coeffs)


def get_OGM_grouping(hamil: QubitOperator):
    hamil, constant = hamil.remove_constant()
    assert constant == 0.0

    covering_pwords = []
    hamil_terms = {k: abs(v) for k, v in hamil.terms.items()}
    ranked_pwords = sorted(hamil_terms, key=hamil_terms.get, reverse=True)
    abs_coeffs = [hamil_terms[pword] for pword in ranked_pwords]
    added_terms = set()
    groups = []
    group_mapping = [set() for pword in ranked_pwords]
    initial_probs = []
    for i in range(len(ranked_pwords)):
        if i in added_terms:
            continue
        covering_pword = ranked_pwords[i]
        group_mapping[i].add(len(groups))
        group = {ranked_pwords[i]}
        initial_prob = 0.0
        for j in chain(range(i + 1, len(ranked_pwords)), range(i)):
            if is_qwc(covering_pword, ranked_pwords[j]):
                covering_pword = get_covering_pauliword([covering_pword, ranked_pwords[j]])
                added_terms.add(j)
                group.add(ranked_pwords[j])
                initial_prob += abs_coeffs[j]
                group_mapping[j].add(len(groups))
        covering_pwords.append(covering_pword)
        groups.append(group)
        initial_probs.append(initial_prob)
    initial_probs = np.array(initial_probs)
    initial_probs = initial_probs / np.sum(initial_probs)

    return GeneralGroupingInfo(group_mapping, ranked_pwords, covering_pwords, initial_probs, abs_coeffs)


if __name__ == '__main__':
    from chemistry.simple_mols import large_12_qubit_lih

    hamil = large_12_qubit_lih()
    grouping_info_ = get_OGM_grouping(hamil)
