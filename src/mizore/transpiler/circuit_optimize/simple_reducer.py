from typing import Iterable, List

from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.meta_circuit.post_processor import SimpleProcessor
from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.quantum_circuit import QuantumCircuit
from mizore.comp_graph.comp_graph import GraphIterator

from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.transpiler.transpiler import Transpiler


class SimpleReducer(Transpiler):

    def transpile(self, target_nodes: GraphIterator):
        node: QCircuitNode
        for node in target_nodes.by_type(QCircuitNode):
            circuit: MetaCircuit = node.circuit
            circuit.add_post_process(ReduceProcessor())

class ReduceProcessor(SimpleProcessor):
    name = "gate_reducer"

    def process(self, gates, block):
        new_gates = []
        for gate in gates:
            reduced_gates = gate.simple_reduce()
            if reduced_gates is None:
                new_gates.append(gate)
            else:
                new_gates.extend(reduced_gates)
        return new_gates
