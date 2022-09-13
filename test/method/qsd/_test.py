from chemistry.simple_mols import simple_4_qubit_lih, simple_8_qubit_h4
from mizore.backend_circuit.one_qubit_gates import X
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.qsd.krylov_single_ref import quantum_krylov_single_ref
from mizore.method.qsd.krylov_single_ref_classical import quantum_krylov_single_ref_classical
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
import jax

from mizore.transpiler.estimator.simple_resource import SimpleResource
from mizore.transpiler.hardware_config.hardware_config_example import example_superconducting_circuit_config
from mizore.transpiler.measurement.l1 import L1Sampling
from mizore.transpiler.replace.trotterizer import Trotterizer

jax.config.update("jax_enable_x64", True)

hamil, init_circuit = simple_8_qubit_h4()

n_basis = 4
delta = 1.0
realistic = True
eigv_classical, H_classical, S_classical = quantum_krylov_single_ref_classical(init_circuit, hamil, n_basis, delta, get_H_S=True)
print(eigv_classical)

eigvals, H, S = quantum_krylov_single_ref(init_circuit, hamil, n_basis, delta, eps=1e-11, get_H_S=True)
cg = eigvals.build_graph()

CircuitRunner(state_processor_gens=[L1Sampling(state_ignorant=False)]) | cg

for node in cg.all():
    node.shot_num.set_value(1e7)


eigvals.show_value()
eigvals.show_std_devi()

if realistic:
    Trotterizer(0.1) | cg
    SimpleReducer() | cg

if not realistic:
    exit(0)

resource = SimpleResource(example_superconducting_circuit_config()) | cg.all()

total_time = 0
for node_dict in resource.values():
    total_time += node_dict["total_time"]
total_time = total_time * 1e-6

print("total time(h): ", total_time/3600)
