from typing import Iterable

from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.comp_node import CompNode
from mizore.comp_graph.node.mc_node import MetaCircuitNode

from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.transpiler.estimator.utils.count_gates import count_one_two_qubit_gates
from mizore.transpiler.transpiler import Transpiler
from mizore.transpiler.hardware_config.simple_hardware import SimpleHardware


class SimpleResource(Transpiler):
    def __init__(self, hardware: SimpleHardware):
        super().__init__()
        self.hardware = hardware

    def transpile(self, target_nodes: GraphIterator):
        hardware = self.hardware
        output_dict = {}
        node: QCircuitNode
        for node in target_nodes.by_type(QCircuitNode):
            res = {}
            if isinstance(node, MetaCircuitNode):
                n_one, n_two = count_one_two_qubit_gates(node.circuit.get_gates(node.params.mean.value()))
            else:
                n_one, n_two = count_one_two_qubit_gates(node.circuit.get_gates())
            res["n_one_gate"] = n_one
            res["n_two_gate"] = n_two
            res["n_gate"] = n_one + n_two
            fidelity = 1.0
            fidelity *= (1 - hardware.two_qubit_gate_error) ** n_two
            fidelity *= (1 - hardware.one_qubit_gate_error) ** n_one
            fidelity *= (1 - hardware.readout_error)
            res["fidelity"] = fidelity

            time = 0
            time += hardware.init_time
            time += n_one * hardware.one_qubit_gate_time
            time += n_two * hardware.two_qubit_gate_time
            time += hardware.readout_time
            res["time"] = time
            try:
                res["total_time"] = float(node.shot_num.value()*time)
            except Exception:
                pass

            output_dict[node] = res

        return output_dict
