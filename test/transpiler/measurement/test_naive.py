from circuit_utils.sample_pqc_node import simple_pqc_node
from mizore.comp_graph.comp_graph import CompGraph
from mizore.transpiler.measurement.naive import NaiveMeasurement


def test_pqc_node():
    expv = simple_pqc_node()()
    cg = CompGraph([expv.mean, expv.var])
    NaiveMeasurement() | cg
    expv.var.show_value()
    assert True
