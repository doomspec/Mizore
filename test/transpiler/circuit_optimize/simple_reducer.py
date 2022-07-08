from math import pi

from mizore.backend_circuit.controlled import ControlledGate
from mizore.backend_circuit.multi_qubit_gates import PauliGate
from mizore.backend_circuit.one_qubit_gates import GlobalPhase, Hadamard
from mizore.backend_circuit.rotations import SingleRotation
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.meta_circuit.block.controlled import Controlled
from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner

if __name__ == '__main__':
    n_qubit = 2 # , GlobalPhase(-pi / 2)
    blocks = [GateGroup(Hadamard(0), Hadamard(1)), Controlled(GateGroup(GlobalPhase(1.5)), [1])]
    #blocks = [GateGroup(Hadamard(1), ControlledGate(SingleRotation(1, 0, 0.1), 1), GlobalPhase(1.0))]
    circuit = MetaCircuit(n_qubit, blocks)
    obs = [QubitOperator(f"{op}1") for op in ["X", "Y", "Z"]]
    node = QCircuitNode(circuit, obs)
    expv = node()
    CircuitRunner() | node
    expv.show_value()
    expv.del_cache_recursive()
    print(circuit)
    SimpleReducer() | node
    CircuitRunner() | node
    expv.show_value()
    print(circuit)


