from circuit_utils.sample_pqc_node import simple_pqc_node, simple_large_pqc_node
from mizore.comp_graph.comp_graph import CompGraph
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.noise_model.simple_noise import DepolarizingNoise

output = []
for i in range(80):
    pqc_node = simple_large_pqc_node()
    output.append(pqc_node().mean)
    pqc_node.circuit.has_random = True
cg = CompGraph(output)
SimpleReducer() | cg
DepolarizingNoise(0.01) | cg
CircuitRunner(n_proc=2) | cg
