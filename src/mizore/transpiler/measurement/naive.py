
from mizore.backend_circuit.backend_op import BackendOperator
from mizore.backend_circuit.backend_state import BackendState
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.state_processor import StateProcessor
from mizore.transpiler.transpiler import Transpiler
from mizore import jax_array
from math import sqrt

# Var = 4(1-p)p for two-point distribution where probability
# of being 1 is p and being -1 is (1-p);
# We set here unit_variance = max Var = 4*0.5*0.5 = 1
unit_variance = 1.0


class NaiveMeasurement(Transpiler):

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
            return NaiveMeasurementStateProcessor(node, state_ignorant=self.state_ignorant)
        else:
            return None


class NaiveMeasurementStateProcessor(StateProcessor):
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
        weight_sum = sum([weight for _, weight in ob.terms.items()])
        var_coeffs.append(weight_sum ** 2 * unit_variance)
    return var_coeffs


def process_state_aware(obs_list, state: BackendState):
    var_coeffs = []
    for ob in obs_list:
        weight_list = []
        for qset, op, weight in ob.qset_op_weight():
            if len(qset) == 0:
                continue
            backend_op = BackendOperator(QubitOperator.from_qset_op(qset, op))
            prob = (backend_op.get_expectation_value(state) + 1) / 2
            assert prob.imag < 1e-8  # In case the imaginary part is not zero, there must be a bug
            prob = prob.real
            var = 4 * (1 - prob) * prob  # variance of two-point distribution
            if var < 0:
                var += 1e-14
            weight_list.append(abs(weight) * sqrt(var))
        weight_sum = sum(weight_list)
        var_coeffs.append(weight_sum ** 2)
    return var_coeffs



"""
def get_qc_node_var_coeff(node: QCircuitNode, ob):
    # TODO here density matrix is not considered. I don't know when to use dm
    state = node.circuit.get_backend_circuit().get_quantum_state()
    weight_list = []  # contains a_i*sqrt(Var(<O_i>)) for H=a_iO_i
    for qset, op, weight in ob.qset_op_weight():
        backend_op = BackendOperator(QubitOperator.from_qset_op(qset, op))
        # TODO check this. Maybe write a test
        # The probability of getting 1.
        # (1-prob) is the probability of getting -1
        # In the state ignorant mode, prob is assumed to be 0.5 to maximize the variance
        prob = (backend_op.get_expectation_value(state) + 1) / 2
        assert prob.imag < 1e-8  # In case the imaginary part is not zero, there must be a bug
        prob = prob.real
        var = 4 * (1 - prob) * prob  # variance of two-point distribution
        if var < 0:
            var += 1e-14
        weight_list.append(abs(weight) * sqrt(var))
    del state
    weight_sum = sum(weight_list)
    return weight_sum ** 2
"""