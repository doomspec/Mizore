from mizore.comp_graph.comp_graph import GraphIterator
from mizore.transpiler.transpiler import Transpiler
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.measurement.grouping_utils.qwc import get_qwc_cliques_by_LDF, get_shot_ratio_from_groupings
import numpy as np

class GroupingMeasurement(Transpiler):
    def __init__(self, method="LDF"):
        super().__init__()
        self.method = method

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            if "GroupingMeasurement" not in node.aux_info_dict:
                raise Exception()
            state_ignorant = False
            if "GroupingMeasurement" not in node.aux_obs_dict:
                state_ignorant = True
            aux_info_dict = node.aux_info_dict["GroupingMeasurement"]
            coeff_list = np.array(aux_info_dict["coeff_list"])
            if not state_ignorant:
                op_prod_expv_list = np.array(node.aux_obs_dict["GroupingMeasurement"]["res"])
                op_prod_expv_list = op_prod_expv_list**aux_info_dict["n_support_list"]
            else:
                op_prod_expv_list = np.array([1.0]*len(coeff_list))
            var_coeff = np.sum(coeff_list*op_prod_expv_list)
            var_coeff -= (node.expv - node.obs.constant) ** 2
            node.expv.set_to_random_variable(var_coeff / node.shot_num, check_valid=False)

    @classmethod
    def prepare(cls, method="LDF", state_ignorant=False):
        return GroupingMeasurementPrep(method=method, state_ignorant=state_ignorant)

class GroupingMeasurementPrep(Transpiler):
    def __init__(self, method, state_ignorant):
        super().__init__()
        self.method = method
        self.state_ignorant = state_ignorant
        self.grouping_cache = {}

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            obs = node.obs
            if obs in self.grouping_cache:
                grouping = self.grouping_cache[obs]
            else:
                grouping = get_qwc_cliques_by_LDF(obs)
                self.grouping_cache[obs] = grouping
            group_shot_ratios = get_shot_ratio_from_groupings(grouping, obs)

            op_prod_list = []
            coeff_list = []
            n_support_list = []
            for i_group in range(len(grouping)):
                #print(grouping[i_group])
                weight_list = []
                sub_ops = []
                for op_tuple in grouping[i_group]:
                    op = QubitOperator.from_op_tuple(op_tuple)
                    sub_ops.append(op)
                    weight = obs.terms[op_tuple]
                    weight_list.append(weight)
                if not self.state_ignorant:
                    for i in range(len(sub_ops)):
                        for j in range(len(sub_ops)):
                            op_prod = sub_ops[i] * sub_ops[j]
                            op_tuple, _ = op_prod.get_unique_op_tuple()
                            n_support = len(op_tuple)
                            #print(repr(sub_ops[i]), repr(sub_ops[j]), sub_ops[i] * sub_ops[j])
                            # if n_support == 0:
                            #    continue
                            op_prod_list.append(op_prod)
                            coeff_list.append(weight_list[i] * weight_list[j] / group_shot_ratios[i_group])
                            n_support_list.append(n_support)
                else:
                    for i in range(len(sub_ops)):
                        for j in range(len(sub_ops)):
                            coeff_list.append(weight_list[i] * weight_list[j] / group_shot_ratios[i_group])
            node.aux_info_dict["GroupingMeasurement"] = {"coeff_list": coeff_list}
            if not self.state_ignorant:
                node.aux_obs_dict["GroupingMeasurement"] = {"obs": op_prod_list}
                node.aux_info_dict["GroupingMeasurement"]["n_support_list"] = n_support_list
