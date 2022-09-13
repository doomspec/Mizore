from chemistry.simple_mols import simple_4_qubit_lih, simple_8_qubit_h4
from circuit_utils.sample_circuit import circuit_for_test_0
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.measurement.grouping import GroupingMeasurement
from mizore.transpiler.measurement.l1 import L1Sampling, L1Allocation

shot_num = 100
hamil, circuit = simple_8_qubit_h4()
hamil, _ = hamil.remove_constant()
hamil_weight = hamil.get_l1_norm_omit_const()
circuit = MetaCircuit(8)
node0 = DeviceCircuitNode(circuit, hamil)
node0.shot_num.set_value(shot_num)
GroupingMeasurement.prepare(state_ignorant=False) | node0
CircuitRunner() | node0
GroupingMeasurement() | node0
expv0 = node0()
mean0 = expv0.value()
var0 = expv0.var.value()


node1 = DeviceCircuitNode(circuit, hamil)
node1.shot_num.set_value(shot_num)
#L1Allocation.prepare() | node1
CircuitRunner() | node1
L1Sampling(state_ignorant=False) | node1
expv1 = node1()
mean1 = expv1.value()
var1 = expv1.var.value()


print("other",var1)
print("group",var0)