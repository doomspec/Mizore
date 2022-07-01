from typing import List

from mizore.meta_circuit.block import Block
from mizore.meta_circuit.post_processor import SimpleProcessor
from mizore.comp_graph.comp_graph import GraphIterator

from mizore.backend_circuit.noise import Depolarizing
from mizore.meta_circuit.meta_circuit import MetaCircuit

from mizore.backend_circuit.quantum_circuit import QuantumCircuit

from mizore.backend_circuit.gate import Gate
from mizore.comp_graph.node.mc_node import MetaCircuitNode

from mizore.comp_graph.node.qc_node import QCircuitNode

from mizore.transpiler.transpiler import Transpiler


# See https://qiskit.org/documentation/apidoc/aer_noise.html

class DepolarizingNoise(Transpiler):
    """
    A transpiler for add
    """

    def __init__(self, error_rate=None, two_qubit_rate=None):
        super().__init__()
        assert error_rate > 0
        self.error_rate = error_rate
        if two_qubit_rate is None:
            self.two_qubit_rate = error_rate

    def transpile(self, graph_iterator: GraphIterator):
        mcnode: MetaCircuitNode

        for mcnode in graph_iterator.by_type(MetaCircuitNode):
            noisy_circuit: MetaCircuit = mcnode.circuit
            # Add noise make to the backend_circuit post processor
            noisy_circuit.add_post_process(SimpleNoiseMaker(self.error_rate, self.two_qubit_rate))
            noisy_circuit.has_random = True

            mcnode.tags.add("noisy")
            mcnode.name = mcnode.name + "-Noisy"


class SimpleNoiseMaker(SimpleProcessor):
    name = "noise_maker"

    def __init__(self, error_rate, two_qubit_rate):
        self.error_rate = error_rate
        self.two_qubit_rate = two_qubit_rate

    def process(self, gates, block: Block):
        if block.attr.get("no_noise", False):
            return gates
        noisy = []
        for gate in gates:
            n_qubit = len(gate.qset)
            if n_qubit > 2 and not gate.is_noise:
                raise Exception("Circuit contains a gate acts on more than two gate. Consider reduce the gates.")
            noisy.append(gate)
            if gate.is_noise:
                continue
            if n_qubit == 1:
                noisy.append(Depolarizing(gate.qset[0], self.error_rate))
            if n_qubit == 2:
                noisy.append(Depolarizing(gate.qset[0], self.two_qubit_rate))
                noisy.append(Depolarizing(gate.qset[1], self.two_qubit_rate))
        return noisy
