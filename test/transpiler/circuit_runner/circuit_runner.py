from circuit_utils.sample_pqc_node import simple_pqc_node
from circuit_utils.sample_qc_node import simple_qc_node
from mizore.comp_graph.comp_graph import CompGraph
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.measurement.naive import NaiveMeasurement

#node0 = simple_pqc_node(param_var=0.0)
node1 = simple_pqc_node(param_var=0.001)
#exp_valvar0 = node0()
exp_valvar1 = node1()
cg = CompGraph([exp_valvar1])
CircuitRunner() | cg
NaiveMeasurement() | cg

#node1.params.show_value()

exp_valvar1.show_value()
