

from mizore import to_jax_array
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.transpiler.transpiler import Transpiler

available_method = {"full_clifford", "mixed_clifford", "random_pauli"}

class CliffordShadowEst(Transpiler):

    def __init__(self, method, name=None, clifford_groups=None):
        Transpiler.__init__(self, name)
        assert method in available_method
        self.method = method
        if method == "mixed_clifford":
            assert clifford_groups is not None


    def transpile(self, graph_iterator: GraphIterator):
        node: MetaCircuitNode
        for node in graph_iterator.by_type(MetaCircuitNode):
            node.exp_var.set_value(to_jax_array([0.0]*len(node.obs)))