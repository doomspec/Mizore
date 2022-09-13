from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.parameter import Parameter
from mizore.transpiler.transpiler import Transpiler
import numpy as np


class L1Sampling(Transpiler):

    def __init__(self, state_ignorant=False, name=None, default_shot_num=10000):
        Transpiler.__init__(self, name)
        self.state_ignorant = state_ignorant
        self.default_shot_num = default_shot_num

    def transpile(self, graph_iterator: GraphIterator):
        output_dict = {}
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            if node.shot_num.value() == -1:
                node.shot_num.bind_to(Parameter(self.default_shot_num))
            var_coeff = (node.obs.get_l1_norm_omit_const()) ** 2
            if not self.state_ignorant:
                var_coeff -= (node.expv - node.obs.constant) ** 2
            node.expv.set_to_random_variable(var_coeff / node.shot_num, check_valid=False)
        return output_dict


class L1Allocation(Transpiler):

    def __init__(self, state_ignorant=False, name=None, default_shot_num=10000):
        Transpiler.__init__(self, name)
        self.state_ignorant = state_ignorant
        self.default_shot_num = default_shot_num

    def transpile(self, graph_iterator: GraphIterator):
        output_dict = {}
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            if node.shot_num.value() == -1:
                node.shot_num.bind_to(Parameter(self.default_shot_num))
            if self.state_ignorant:
                var_coeff = (node.obs.get_l1_norm_omit_const()) ** 2
            else:
                if "L1Allocation" not in node.aux_obs_dict:
                    raise Exception("Have you run L1AllocationPrep?")
                if "res" not in node.aux_obs_dict["L1Allocation"]:
                    raise Exception("Have you run the circuit runner?")
                aux_obs_dict = node.aux_obs_dict["L1Allocation"]
                weight_list = np.abs(np.array(aux_obs_dict["aux_weight_list"]))
                expvs = np.array(aux_obs_dict["res"])
                var_coeff = np.sum(weight_list * (np.sqrt(1 - expvs ** 2))) ** 2
            node.expv.set_to_random_variable(var_coeff / node.shot_num, check_valid=False)
        return output_dict

    @classmethod
    def prepare(cls):
        return L1AllocationPrep()


class L1AllocationPrep(Transpiler):

    def __init__(self):
        Transpiler.__init__(self, None)

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            aux_dict = {}
            obs_list = []
            weight_list = []
            for pauliword, weight in node.obs.iter_sub_ops():
                obs_list.append(pauliword)
                weight_list.append(weight)
            aux_dict["obs"] = obs_list
            aux_dict["aux_weight_list"] = weight_list
            node.aux_obs_dict["L1Allocation"] = aux_dict
