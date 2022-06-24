from qulacs import QuantumState
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.operators.qulacs import iter_qulacs_ops
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
        node: MetaCircuitNode
        for node in graph_iterator.by_type(MetaCircuitNode):
            # circuit_output = {}
            var_coeffs = []
            if not self.state_ignorant:  # The state aware mode
                for ob in node.obs:
                    var_coeffs.append(get_qc_node_var_coeff(node, ob))
            else:  # The state ignorant mode
                for ob in node.obs:
                    weight_sum = sum([weight for _, weight in ob.operator.terms.items()])
                    var_coeffs.append(weight_sum ** 2 * unit_variance)
            var_coeffs = jax_array(var_coeffs)
            node.exp_var.set_value(var_coeffs / node.shot_num)
        return output_dict


def get_qc_node_var_coeff(node: QCircuitNode, ob):
    circuit = node.circuit.get_backend_circuit()
    state = QuantumState(node.circuit.n_qubit)
    circuit.update_quantum_state(state)
    weight_list = []  # contains a_i*sqrt(Var(<O_i>)) for H=a_iO_i
    for qulacs_op, weight in iter_qulacs_ops(ob.operator, node.circuit.n_qubit):
        # TODO check this
        # The probability of getting 1.
        # (1-prob) is the probability of getting -1
        # In the state ignorant mode, prob is assumed to be 0.5 to maximize the variance
        prob = (qulacs_op.get_expectation_value(state) + 1) / 2
        assert prob.imag < 1e-8  # In case the imaginary part is not zero, there must be a bug
        prob = prob.real
        var = 4 * (1 - prob) * prob  # variance of two-point distribution
        weight_list.append(abs(weight) * sqrt(var))
    del state
    weight_sum = sum(weight_list)
    return weight_sum ** 2
