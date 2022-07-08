from circuit_utils.sample_qc_node import simple_qc_node
from mizore.comp_graph.comp_graph import CompGraph
from mizore.transpiler.measurement.naive import NaiveMeasurement

expv = simple_qc_node()()
print(expv.name)
cg = CompGraph([expv])
NaiveMeasurement(state_ignorant=True) | cg
print(expv.name)