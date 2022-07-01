from circuit_utils.sample_qc_node import simple_qc_node
from mizore.comp_graph.comp_graph import CompGraph
from mizore.transpiler.measurement.naive import NaiveMeasurement

exp_valvar = simple_qc_node()()
cg = CompGraph([exp_valvar.mean, exp_valvar.var])
NaiveMeasurement(state_ignorant=True) | cg
exp_valvar.var.show_value()