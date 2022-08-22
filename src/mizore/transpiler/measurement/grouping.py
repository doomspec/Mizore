from mizore.backend_circuit.backend_op import BackendOperator
from mizore.backend_circuit.backend_state import BackendState
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.state_processor import StateProcessor
from mizore.transpiler.measurement.grouping.qwc import get_qwc_cliques_by_LDF
from mizore.transpiler.measurement.l1sampling import L1Sampling, get_var_list
from mizore import jax_array
from mizore.transpiler.measurement.utils import unit_variance


class GroupingMeasurement(L1Sampling):
    def __init__(self, method="LDF", state_ignorant=False):
        super().__init__(state_ignorant=state_ignorant)
        self.method = method
        self.grouping_cache = {}

    def get_state_processor(self, node):
        if isinstance(node, DeviceCircuitNode):
            groupings = []
            for ob in node.obs_list:
                if ob in self.grouping_cache.keys():
                    groupings.append(self.grouping_cache[ob])
                else:
                    grouping = get_qwc_cliques_by_LDF(ob)
                    self.grouping_cache[ob] = grouping
                    groupings.append(grouping)
            return GroupingMeasurementStateProcessor(node, state_ignorant=self.state_ignorant)
        else:
            return None


class GroupingMeasurementStateProcessor(StateProcessor):

    def __init__(self, node, grouping_list, state_ignorant=False):
        super().__init__("GroupingMeasurementVarCoeff")
        self._weight_processor = process_state_ignorant if state_ignorant else process_state_aware
        self.obs_list = node.obs_list
        self.grouping_list = grouping_list

    def process(self, state: BackendState) -> any:
        return self._weight_processor(self.obs_list, self.grouping_list, state)

    def post_process(self, node, process_res):
        if not isinstance(node, DeviceCircuitNode):
            return
        var_coeffs = jax_array(process_res) if not node.is_single_obs else process_res[0]
        node.expv.set_to_random_variable(var_coeffs / node.shot_num, check_valid=False)


def process_state_ignorant(obs_list, grouping_list, state: BackendState):
    var_coeffs = []
    for ob in obs_list:
        weight_sum = sum([abs(weight) for _, weight in ob.terms.items()])
        var_coeffs.append(weight_sum ** 2 * unit_variance)
    return var_coeffs


def process_state_aware(obs_list, grouping_list, state: BackendState):
    var_coeffs = []
    for ob in obs_list:
        var_list = get_var_list(ob, state)
        weight_sum = sum(weight_list)
        var_coeffs.append(weight_sum ** 2)
    return var_coeffs
