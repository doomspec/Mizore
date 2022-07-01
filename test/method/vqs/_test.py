import jax.numpy.linalg

from chemistry.simple_mols import simple_4_qubit_lih
from method.vqs.circuits import diff_inner_product, A_mat, C_mat
from mizore.comp_graph.calc_node.linear_sys_node import LinearSysNode
from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.comp_graph.valvar import ValVar
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
import numpy as np

from mizore.transpiler.estimator.simple_resource import SimpleResource
from mizore.transpiler.hardware_config.hardware_config_example import example_superconducting_circuit_config
from mizore.transpiler.measurement.infinite import InfiniteMeasurement
from mizore.transpiler.measurement.naive import NaiveMeasurement
from mizore.transpiler.noise_model.simple_noise import DepolarizingNoise

hamil = simple_4_qubit_lih()

blocks = []

i=0
for qset_ops_weight in hamil.qset_ops_weight():
    if len(qset_ops_weight[0]) == 0:
        continue
    blocks.append(Rotation(qset_ops_weight[0],qset_ops_weight[1],qset_ops_weight[2],fixed_angle_shift=1.0))
    i+=1

circuit = MetaCircuit(4, blocks=blocks)

#print(circuit)

A = A_mat(circuit)
C = C_mat(circuit, hamil)

cg = CompGraph([A, C])

for node in cg.by_type(MetaCircuitNode):
    node: MetaCircuitNode
    node.shot_num.set_value(1000)
    node.random_config = {"use_dm": True}


#SimpleReducer() | cg # Here is a bug

NaiveMeasurement() | cg
CircuitRunner() | cg

linear_node = LinearSysNode(A, C)
linear_node().show_value()



#DepolarizingNoise(error_rate=0.000001) | cg
CircuitRunner() | cg
linear_node().show_value()


exit()
resource = SimpleResource(example_superconducting_circuit_config()) | cg.all()

total_time = 0
for node_dict in resource.values():
    total_time += node_dict["total_time"]
total_time = total_time * 1e-6

print("total time(s): ", total_time)