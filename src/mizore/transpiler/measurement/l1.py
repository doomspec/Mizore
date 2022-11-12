from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.parameter import Parameter
from mizore.operators import QubitOperator
from mizore.transpiler.measurement.abst_measure import MeasureImpl, MeasureTranspiler
from mizore.transpiler.transpiler import Transpiler
import numpy as np


class L1SamplingImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot):
        super().__init__(observable, n_shot)
        self.prob = np.abs(np.array(list(self.ob.terms.values())))
        self.weight = np.sum(self.prob)
        self.prob /= self.weight
        self.pauliwords = list(self.ob.terms.keys())

    def get_pauliwords(self):
        sampled_indices = np.random.choice(len(self.pauliwords), self.n_shot, p=self.prob)
        return [self.pauliwords[i] for i in sampled_indices]

    def one_shot_estimation(self, pauliword, qset_for_1):
        return self.weight * (-1) ** len(qset_for_1)


class L1Sampling(MeasureTranspiler):
    def measure_impl(self, observable: QubitOperator, n_shot):
        return L1SamplingImpl(observable, n_shot)

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


class L1AllocationImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot):
        super().__init__(observable, n_shot)
        self.pauliwords = list(self.ob.terms.keys())
        self.weight = np.abs(np.array(list(self.ob.terms.values())))
        self.weight /= np.sum(self.weight)
        self.weight *= n_shot
        self.weight = np.ceil(self.weight)
        self.allocation_dict = {self.pauliwords[i]: int(self.weight[i])
                                for i in range(len(self.pauliwords)) if self.weight[i] != 0}

    def estimate_by_state(self, state):
        occur_dict = self.allocation_dict
        estimation = 0.0
        i = 0
        for pauliword, n_occur in occur_dict.items():
            sampled_qsets = state.sample_pauli_measure_by_pauliword_tuple(pauliword, n_occur)
            i += 1
            coeff = self.ob.terms[pauliword]
            res = coeff * sum([(-1) ** len(qset) for qset in sampled_qsets]) / len(sampled_qsets)
            estimation += res
        return estimation


class L1Allocation(MeasureTranspiler):
    def prepare(self):
        return L1AllocationPrep()

    def measure_impl(self, observable: QubitOperator, n_shot):
        return L1AllocationImpl(observable, n_shot)

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
