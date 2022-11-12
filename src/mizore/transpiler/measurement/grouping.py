from mizore.comp_graph.comp_graph import GraphIterator
from mizore.operators.qubit_operator import PauliTuple
from mizore.transpiler.measurement.abst_measure import MeasureTranspiler, MeasureImpl
from mizore.transpiler.transpiler import Transpiler
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.measurement.grouping_utils.qwc import get_qwc_cliques_by_LDF, get_prob_from_groupings, \
    get_covering_pauliword
import numpy as np


def generate_grouping(method, obs):
    if method == "LDF":
        return get_qwc_cliques_by_LDF(obs)


class GroupedSamplingPrep(Transpiler):
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
            group_probs = get_prob_from_groupings(grouping, obs)

            op_prod_list = []
            coeff_list = []
            n_support_list = []
            node.aux_info_dict["GroupedSampling"] = {"coeff_list": coeff_list}
            for i_group in range(len(grouping)):
                weight_list = []
                sub_ops = []
                for pauliword in grouping[i_group]:
                    op = QubitOperator.from_pauli_tuple(pauliword)
                    sub_ops.append(op)
                    weight = obs.terms[pauliword]
                    weight_list.append(weight)
                if self.cov_approx == "exact":
                    for i in range(len(sub_ops)):
                        for j in range(i, len(sub_ops)):
                            op_prod = sub_ops[i] * sub_ops[j]
                            pauliword, _ = op_prod.get_unique_op_tuple()
                            n_support = len(pauliword)
                            # print(repr(sub_ops[i]), repr(sub_ops[j]), sub_ops[i] * sub_ops[j])
                            # if n_support == 0:
                            #    continue
                            op_prod_list.append(op_prod)
                            coeff = weight_list[i] * weight_list[j] / group_probs[i_group]
                            if i != j:
                                coeff *= 2  # Count the off-diagonal terms twice
                            coeff_list.append(coeff)
                            n_support_list.append(n_support)
                    node.aux_info_dict["GroupedSampling"]["n_support_list"] = n_support_list
                elif self.cov_approx == "diagonal":
                    # We assume Cov(i,j) = 0, and only consider the diagonal terms
                    for i in range(len(sub_ops)):
                        op_prod_list.append(sub_ops[i])
                        coeff_list.append(weight_list[i] * weight_list[i] / group_probs[i_group])
                elif self.cov_approx == "average":
                    # We assume Cov(i,j) = 0, and assume the diagonal terms Tr(P\rho) = 1.0
                    for i in range(len(sub_ops)):
                        coeff_list.append(weight_list[i] * weight_list[i] / group_probs[i_group])
                else:
                    raise Exception()
            """
            coeff_list = [alpha_i * alpha_j / group_prob]
            """
            if self.cov_approx != "average":
                node.aux_obs_dict["GroupedSampling"] = {"obs": op_prod_list}


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


class GroupedSamplingImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot, groupings):
        super().__init__(observable, n_shot)
        self.groupings = groupings
        self.group_probs = get_prob_from_groupings(groupings, observable)
        self.covering_pauliwords = [tuple(get_covering_pauliword(group)) for group in groupings]

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


class GroupedSampling(MeasureTranspiler):
    """
    Implementing the grouped measurement described in
    https://arxiv.org/pdf/2006.15788.pdf
    (Measurements of quantum Hamiltonians with locally-biased classical shadows)
    The variance is estimated by the Equ.30 at Page 14 of the above paper
    """

    def prepare(self):
        return GroupedSamplingPrep(self.method, self.cov_approx, self.grouping_cache)

    def measure_impl(self, observable: QubitOperator, n_shot):
        if observable in self.grouping_cache:
            grouping = self.grouping_cache[observable]
        else:
            grouping = generate_grouping(self.method, observable)
        return GroupedSamplingImpl(observable, n_shot, grouping)

    def __init__(self, method="LDF", cov_approx="exact"):
        """
        Args:
            method:
            cov_approx: options: "exact", "diagonal", "average"
        """
        super().__init__()
        self.method = method
        self.cov_approx = cov_approx
        self.grouping_cache = {}

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            if "GroupedSampling" not in node.aux_info_dict:
                raise Exception()
            cov_approx = self.cov_approx
            if "GroupedSampling" not in node.aux_obs_dict:
                cov_approx = "average"
            aux_info_dict = node.aux_info_dict["GroupedSampling"]
            coeff_list = np.array(aux_info_dict["coeff_list"])
            if cov_approx == "exact":
                op_prod_expv_list = np.array(node.aux_obs_dict["GroupedSampling"]["res"])
                op_prod_expv_list = op_prod_expv_list ** aux_info_dict["n_support_list"]
                var_coeff = np.sum(coeff_list * op_prod_expv_list)
                var_coeff -= (node.expv - node.obs.constant) ** 2
            elif cov_approx == "diagonal":
                op_prod_expv_list = np.array(node.aux_obs_dict["GroupedSampling"]["res"])
                op_prod_var_list = 1.0 - op_prod_expv_list ** 2
                var_coeff = np.sum(np.abs(coeff_list * op_prod_var_list))
            elif cov_approx == "average":
                op_prod_var_list = np.array([1.0] * len(coeff_list))
                var_coeff = np.sum(np.abs(coeff_list * op_prod_var_list))
            else:
                raise Exception()

            node.expv.set_to_random_variable(var_coeff / node.shot_num, check_valid=False)
