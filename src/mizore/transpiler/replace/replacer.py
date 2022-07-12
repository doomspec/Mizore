from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.transpiler.transpiler import Transpiler


class Replacer(Transpiler):
    def __init__(self, only_device_circuit=True):
        super().__init__()
        self.only_device_circuit = only_device_circuit

    def transpile(self, graph_iterator: GraphIterator):
        iterator = graph_iterator.by_type(DeviceCircuitNode) if self.only_device_circuit \
            else graph_iterator.by_type(QCircuitNode)
        node: QCircuitNode
        for node in iterator:
            self.replace(node.circuit)

    def replace(self, circuit: MetaCircuit):
        raise NotImplementedError()


