from mizore.backend_circuit.backend_op import BackendOperator
from mizore.backend_circuit.backend_state import BackendState
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.state_processor import StateProcessor
from mizore.transpiler.measurement.utils import unit_variance
from mizore.transpiler.transpiler import Transpiler
from mizore import jax_array
from math import sqrt


class L1Sampling(Transpiler):

    def __init__(self, state_ignorant=False, name=None):
        Transpiler.__init__(self, name)
        self.state_ignorant = state_ignorant

    def transpile(self, graph_iterator: GraphIterator):
        output_dict = {}
        node: DeviceCircuitNode

        for node in graph_iterator.by_type(DeviceCircuitNode):
            """
            var_coeffs = []
            if not self.state_ignorant:  # The state aware mode
                for ob in node.obs_list:
                    var_coeffs.append(get_qc_node_var_coeff(node, ob))
            else:  # The state ignorant mode
                for ob in node.obs_list:
                    weight_sum = sum([weight for _, weight in ob.terms.items()])
                    var_coeffs.append(weight_sum ** 2 * unit_variance)
            var_coeffs = jax_array(var_coeffs) if not node.is_single_obs else var_coeffs[0]
            node.expv.set_to_random_variable(var_coeffs / node.shot_num, check_valid=False)
            """
            state_processor = self.get_state_processor(node)
            state = node.circuit.get_backend_state(list(node.params.value()))
            res = state_processor.process(state)
            state_processor.post_process(node, res)

        return output_dict

    def get_state_processor(self, node):
        if isinstance(node, DeviceCircuitNode):
            return L1SamplingStateProcessor(node, state_ignorant=self.state_ignorant)
        else:
            return None

class L1SamplingStateProcessor(StateProcessor):
    def __init__(self, node, state_ignorant=False):
        super().__init__("NaiveMeasurementVarCoeff")
        self._weight_processor = process_state_ignorant if state_ignorant else process_state_aware
        self.obs_list = node.obs_list

    def process(self, state: BackendState) -> any:
        return self._weight_processor(self.obs_list, state)

    def post_process(self, node, process_res):
        if not isinstance(node, DeviceCircuitNode):
            return
        var_coeffs = jax_array(process_res) if not node.is_single_obs else process_res[0]
        node.expv.set_to_random_variable(var_coeffs / node.shot_num, check_valid=False)


def process_state_ignorant(obs_list, state: BackendState):
    var_coeffs = []
    for ob in obs_list:
        weight_sum = sum([abs(weight) for _, weight in ob.terms.items()])
        var_coeffs.append(weight_sum ** 2 * unit_variance)
    return var_coeffs


def get_var_list(ob: QubitOperator, state: BackendState):
    var_list = []
    for qset, op, weight in ob.qset_op_weight_omit_const():
        backend_op = BackendOperator(QubitOperator.from_qset_op(qset, op))
        prob = (backend_op.get_expectation_value(state) + 1) / 2
        assert prob.imag < 1e-8  # In case the imaginary part is not zero, there must be a bug
        prob = prob.real
        var = 4 * (1 - prob) * prob  # variance of two-point distribution
        if var < 0:
            var += 1e-14
        # var_list.append(abs(weight) * sqrt(var))
        var_list.append(var)
    return var_list


def process_state_aware(obs_list, state: BackendState):
    var_coeffs = []
    for ob in obs_list:
        var_list = get_var_list(ob, state)
        weight_sum = 0.0
        i = 0
        for qset, op, weight in ob.qset_op_weight_omit_const():
            weight_sum += abs(weight) * sqrt(var_list[i])
            i += 1
        var_coeffs.append(weight_sum ** 2)
    return var_coeffs
