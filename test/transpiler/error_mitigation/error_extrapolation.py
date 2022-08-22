from circuit_utils.sample_pqc_node import simple_pqc_node, simple_large_pqc_node
from mizore.transpiler.hardware_config.hardware_config_example import example_superconducting_circuit_config
from mizore.transpiler.estimator.simple_resource import SimpleResource
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.error_mitigation.error_extrapolation import ErrorExtrapolation
from mizore.transpiler.measurement.infinite import InfiniteMeasurement
from mizore.transpiler.measurement.l1sampling import L1Sampling
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.comp_graph.comp_graph import CompGraph
from mizore.transpiler.noise_model.simple_noise import DepolarizingNoise


qnode = simple_large_pqc_node(param_var = 0.000)
qnode.shot_num.set_value(100000)
qnode.config = {"use_dm": True}
#qnode.config = {"n_exp": 10000}

exp_valvar = qnode()
res = exp_valvar
res.name = "SquareRootExp"
cg = CompGraph([res])

SimpleReducer() | cg
CircuitRunner() | cg
L1Sampling() | cg
true_val = res.value()
print("Val without err", res.value())
print("Variance: ", res.var.value())
cg.del_all_cache()

DepolarizingNoise(0.1) | cg
CircuitRunner() | cg
L1Sampling() | cg
print("Result with noise: ", res.value())
print("Variance: ", res.var.value())
cg.del_all_cache()

ErrorExtrapolation([1.1, 1.2]) | cg

for node in cg.all():
    CircuitRunner() | node
    L1Sampling() | node

print("Result after mitigation: ", res.value())
print("Variance: ", res.var.value())

exit()

n_shots = 1000000
qnode.shot_num.set_value(n_shots)
print("Variance of the observable: ", res.var.value())

SimpleReducer() | cg.all()
resource = SimpleResource(example_superconducting_circuit_config()) | cg.all()

total_time = 0
for node_dict in resource.values():
    total_time += node_dict["total_time"]
total_time = total_time * 1e-6

print("total time(s): ", total_time)