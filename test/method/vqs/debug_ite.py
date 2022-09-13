import math

from jax import grad, jacfwd

from chemistry.simple_mols import simple_4_qubit_lih, simple_4_qubit_lih_1, simple_8_qubit_h4
from mizore.backend_circuit.one_qubit_gates import X
from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block.fixed_block import FixedBlock
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.imag_time_evol import imag_evol_gradient, get_residue_imag_evol
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.estimator.simple_resource import SimpleResource
from mizore.transpiler.hardware_config.hardware_config_example import example_superconducting_circuit_config
from mizore.transpiler.measurement.infinite import InfiniteMeasurement
from mizore.transpiler.measurement.l1 import L1Sampling
import jax.numpy as jnp


#hamil = simple_4_qubit_lih_1() * 1.1
#blocks = [Gates(X(0))]
#hamil = QubitOperator("Z0") + QubitOperator("Z1") + QubitOperator("X1 X2")
# hamil = hamil

hamil, circuit = simple_8_qubit_h4()
blocks = []

for qset_op_weight in hamil.qset_op_weight():
    if len(qset_op_weight[0]) == 0:
        continue
    blocks.append(FixedBlock(Rotation(qset_op_weight[0], qset_op_weight[1], qset_op_weight[2], angle_shift=0.1)))

i = 0
for qset_op_weight in hamil.qset_op_weight():
    if len(qset_op_weight[0]) == 0:
        continue
    blocks.append(Rotation(qset_op_weight[0], qset_op_weight[1], qset_op_weight[2], angle_shift=0.0))
    i += 1
    if i >= 30:
        break


#circuit = MetaCircuit(4, blocks=blocks)
circuit.add_blocks(blocks)
#print(circuit)
param = Value([0.3] * circuit.n_param)
rcond = 1e-2

hamil_sq_op = hamil * hamil
hamil_sq_op.compress()

evol_grad, A, C, curr_energy = imag_evol_gradient(circuit, hamil, param, rcond=rcond)
hamil_sq = QCircuitNode(circuit, hamil_sq_op)()

cg = CompGraph([evol_grad, hamil_sq])

for node in cg.by_type(DeviceCircuitNode):
    node: DeviceCircuitNode
    node.expv_shift_from_var = False
    # node.config = {"use_dm": True}

SimpleReducer() | cg
L1Sampling(state_ignorant=True, default_shot_num=1e3) | cg
CircuitRunner() | cg

evol_grad.show_value()
evol_grad.var.show_value()

print("l1 norm", jnp.linalg.norm(evol_grad.value(), ord=1))
print("C.value", C.value())
print("A.value", A.value())
distance_sqr = evol_grad.dot(Value.binary_operator(A.value(), evol_grad, jnp.dot)) + 2 * evol_grad.dot(C.value()) + (hamil_sq.value() - curr_energy.value() ** 2)
#distance_sqr = evol_grad.dot(Value.binary_operator(A.value(), evol_grad, jnp.dot)) + 2 * evol_grad.vdot(C.value())
#distance_sqr = jnp.sqrt | distance_sqr
print("Delta^2",distance_sqr.value())
print("err",jnp.sqrt(distance_sqr.var.value()))
for sample in distance_sqr.gaussian_sample_iter(10, use_jit=True):
    print(sample)
"""
residue = Value.unary_operator(get_residue_imag_evol(A, C, rcond=rcond), jnp.linalg.norm)
print(jnp.linalg.norm(C.value()))
print(jnp.sqrt(residue.var.value()))
"""

resource = SimpleResource(example_superconducting_circuit_config()) | cg.all()

total_time = 0
for node_dict in resource.values():
    total_time += node_dict["total_time"]
total_time = total_time * 1e-6

print("total time(h): ", total_time / 3600)
