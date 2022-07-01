from mizore.backend_circuit.one_qubit_gates import X
from mizore.meta_circuit.block.gate_group import GateGroup


block = GateGroup(*[X(0), X(1)])
print(block.gates)
block = GateGroup(X(0), X(1))
print(block.gates)