from mizore import to_jax_array
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.transpiler.transpiler import Transpiler


class InfiniteMeasurement(Transpiler):

    def __init__(self, name=None):
        Transpiler.__init__(self, name)

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            node.expv.set_to_not_random()
