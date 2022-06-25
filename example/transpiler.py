from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.transpiler.error_mitigation.error_extrapolation import ErrorExtrapolation
from mizore.transpiler.measurement.infinite import InfiniteMeasurement
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.operators import QubitOperator
from mizore.backend_circuit.rotations import SingleRotation
from mizore.backend_circuit.two_qubit_gates import CNOT
from mizore.comp_graph.comp_graph import CompGraph
from mizore.operators.observable import Observable
from mizore.transpiler.noise_model.simple_noise import DepolarizingNoise

n_qubit = 2
mc = MetaCircuit(n_qubit)
mc.add_gates([SingleRotation(1, 0, 0.5), CNOT(0, 1)])
obs = Observable(n_qubit, QubitOperator('Z0 Z1'))
circuit_node = MetaCircuitNode(mc, obs, "MyCircuit")
exp_valvar = circuit_node()
res = exp_valvar + exp_valvar

cg = CompGraph([res])

output = CircuitRunner() | cg.iter_nodes_by_prefix("MyCircuit")
error = res - res.mean.value()
DepolarizingNoise(0.1) | cg.all()
CircuitRunner() | cg.all()
print("Error with noise: ", abs(error.mean.value()))
ErrorExtrapolation([1.2]) | cg.all()
InfiniteMeasurement() | cg.all()
CircuitRunner() | cg.all()
print("Error after mitigation: ", abs(error.mean.value()))