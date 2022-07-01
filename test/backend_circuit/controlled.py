from math import pi
from qulacs import QuantumState
from mizore.backend_circuit.one_qubit_gates import GlobalPhase, Hadamard
from mizore.meta_circuit.block.controlled import Controlled
from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit


blocks = [GateGroup(Hadamard(1)), Controlled(GateGroup(GlobalPhase(-pi / 2)), (1,))]
#blocks = [GateGroup(GlobalPhase(-pi/2))]
circuit = MetaCircuit(2, blocks=blocks)

state = QuantumState(circuit.n_qubit)
circuit.get_backend_circuit().update_quantum_state(state)

print(state)