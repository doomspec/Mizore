from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.comp_graph.comp_graph import CompGraph
from mizore.operators.observable import Observable
from mizore.operators import QubitOperator
from mizore.backend_circuit.two_qubit_gates import CNOT
from mizore.backend_circuit.rotations import SingleRotation
from mizore.meta_circuit.meta_circuit import MetaCircuit

n_qubit = 2
mc = MetaCircuit(n_qubit)
mc.add_gates([SingleRotation(1, 0, 0.5), CNOT(0, 1)])
obs = Observable(n_qubit, QubitOperator('Z0 Z1'))
mcnode = DeviceCircuitNode(mc, obs, "MyCircuit")
exp_valvar = mcnode()

square_res = exp_valvar*exp_valvar

cg = CompGraph([square_res])
CircuitRunner() | cg.all()

print(square_res.mean.value())
