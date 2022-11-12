from typing import List, Set

from mizore.comp_graph.comp_graph import GraphIterator
from mizore.operators.qubit_operator import PauliTuple
from mizore.transpiler.measurement.abst_measure import MeasureTranspiler, MeasureImpl
from mizore.transpiler.measurement.grouping import GroupedSampling
from mizore.transpiler.measurement.grouping_utils.ogm import get_OGM_grouping, GeneralGroupingInfo
from mizore.transpiler.transpiler import Transpiler
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.measurement.grouping_utils.qwc import get_qwc_cliques_by_LDF, get_prob_from_groupings, \
    get_covering_pauliword
import numpy as np


def generate_grouping(method, obs) -> GeneralGroupingInfo:
    if method == "OGM":
        return get_OGM_grouping(obs)


def get_pword_probs(grouping_info: GeneralGroupingInfo):
    group_mapping = grouping_info.group_mapping
    probs = grouping_info.probs
    pword_prob_list = []
    for i in range(len(group_mapping)):
        group_indices = group_mapping[i]
        pword_prob = sum([probs[j] for j in group_indices])
        pword_prob_list.append(pword_prob)
    return np.array(pword_prob_list)


def get_expv_multiplier(grouping_info: GeneralGroupingInfo):
    group_mapping: List[Set] = grouping_info.group_mapping
    pword_list = grouping_info.ranked_pwords
    ops_list = [QubitOperator.from_pauli_tuple(pword) for pword in pword_list]
    coeff_list = grouping_info.abs_coeffs
    probs = grouping_info.probs
    pword_prob_list = get_pword_probs(grouping_info)
    multiplier_list = []
    ops_to_measure = []
    for i in range(len(pword_list)):
        for j in range(i, len(pword_list)):
            multiplier = coeff_list[i] * coeff_list[j]
            shared_groups = group_mapping[i].intersection(group_mapping[j])
            if len(shared_groups) == 0:
                continue
            joint_prob = sum([probs[j] for j in shared_groups])
            multiplier *= joint_prob / (pword_prob_list[i] * pword_prob_list[j])
            if i != j:
                multiplier *= 2
            prod_pword = ops_list[i] * ops_list[j]
            ops_to_measure.append(prod_pword)
            multiplier_list.append(multiplier)
    return ops_to_measure, multiplier_list


class GeneralGroupingPrep(Transpiler):
    def __init__(self, method, cov_approx, grouping_cache):
        super().__init__()
        self.method = method
        self.cov_approx = cov_approx
        self.grouping_cache = grouping_cache

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            obs = node.obs
            if obs in self.grouping_cache:
                grouping = self.grouping_cache[obs]
            else:
                grouping = generate_grouping(self.method, obs)
                self.grouping_cache[obs] = grouping

            if self.cov_approx == "exact":
                ops_list, multiplier_list = get_expv_multiplier(grouping)
                node.aux_info_dict["GeneralGrouping"] = {"multiplier_list": multiplier_list}
                node.aux_obs_dict["GeneralGrouping"] = {"obs": ops_list}
            elif self.cov_approx == "average" or self.cov_approx == "diagonal":
                multiplier_list = grouping.abs_coeffs ** 2
                multiplier_list = multiplier_list / get_pword_probs(grouping)
                node.aux_info_dict["GeneralGrouping"] = {"multiplier_list": multiplier_list}
                if self.cov_approx == "diagonal":
                    ops_list = [QubitOperator.from_pauli_tuple(pword) for pword in grouping.ranked_pwords]
                    node.aux_obs_dict["GeneralGrouping"] = {"obs": ops_list}


def get_measure_res_by_covering_pauliword(sampled_1_qsets, sub_pauliword: PauliTuple):
    res = 0
    for sampled_1_qset in sampled_1_qsets:
        if len(sampled_1_qset) == 0:
            res += 1
            continue
        n_1_in_qset = 0
        i_sampled_1_qset = 0
        # Find the same indices, using the property that qsets are ordered
        for i_qubit, _ in sub_pauliword:
            while i_sampled_1_qset < len(sampled_1_qset) - 1 and sampled_1_qset[i_sampled_1_qset] < i_qubit:
                i_sampled_1_qset += 1
            if sampled_1_qset[i_sampled_1_qset] == i_qubit:
                n_1_in_qset += 1
        res += (-1) ** n_1_in_qset
    return res


class GeneralGroupingImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot, grouping_info):
        super().__init__(observable, n_shot)
        # self.group_probs = grouping_info[]
        # self.covering_pauliwords = [tuple(get_covering_pauliword(group)) for group in groupings]

    def estimate_by_state(self, state):
        group_indices = np.random.choice(len(self.groupings), self.n_shot, p=self.group_probs)
        occur_list = [0 for i in self.groupings]
        for i in group_indices:
            occur_list[i] += 1
        estimation = 0.0
        for i_group in range(len(self.groupings)):
            n_occur = occur_list[i_group]
            sampled_qsets = state.sample_pauli_measure_by_pauliword_tuple(self.covering_pauliwords[i_group], n_occur)
            coeffs = [self.ob.terms[pauliword] for pauliword in self.groupings[i_group]]
            sub_pauliword_res = [get_measure_res_by_covering_pauliword(sampled_qsets, pauliword) for pauliword in
                                 self.groupings[i_group]]
            estimation += sum([coeffs[i] * sub_pauliword_res[i] for i in range(len(coeffs))]) / self.group_probs[
                i_group]
        return estimation / self.n_shot


class GeneralGrouping(GroupedSampling):
    """
    Implementing the grouped measurement described in
    https://arxiv.org/pdf/2006.15788.pdf
    (Measurements of quantum Hamiltonians with locally-biased classical shadows)
    The variance is estimated by the Equ.30 at Page 14 of the above paper
    """

    def prepare(self):
        return GeneralGroupingPrep(self.method, self.cov_approx, self.grouping_cache)

    def measure_impl(self, observable: QubitOperator, n_shot):
        if observable in self.grouping_cache:
            grouping = self.grouping_cache[observable]
        else:
            grouping = generate_grouping(self.method, observable)
        return GeneralGroupingImpl(observable, n_shot, grouping)

    def __init__(self, method="OGM", cov_approx="exact"):
        """
        Args:
            method:
            cov_approx: options: "exact", "diagonal", "average"
        """
        super().__init__(method=method, cov_approx=cov_approx)

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            if "GeneralGrouping" not in node.aux_info_dict:
                raise Exception()
            cov_approx = self.cov_approx
            if "GeneralGrouping" not in node.aux_obs_dict:
                cov_approx = "average"
            aux_info_dict = node.aux_info_dict["GeneralGrouping"]
            multiplier_list = np.array(aux_info_dict["multiplier_list"])
            if cov_approx == "exact":
                op_prod_expv_list = np.array(node.aux_obs_dict["GeneralGrouping"]["res"])
                print(sum(multiplier_list), sum(op_prod_expv_list), len(op_prod_expv_list))
                var_coeff = np.sum(multiplier_list * op_prod_expv_list)
                var_coeff -= (node.expv - node.obs.constant) ** 2
            elif cov_approx == "diagonal":
                op_prod_expv_list = np.array(node.aux_obs_dict["GeneralGrouping"]["res"])
                op_prod_var_list = 1.0 - op_prod_expv_list ** 2
                var_coeff = np.sum(np.abs(multiplier_list * op_prod_var_list))
            elif cov_approx == "average":
                var_coeff = np.sum(multiplier_list)
            else:
                raise Exception()

            node.expv.set_to_random_variable(var_coeff / node.shot_num, check_valid=False)
