from time import time

from chemistry.simple_mols import simple_4_qubit_lih
from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.backend_circuit.one_qubit_gates import X
from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.comp_param import CompParam
from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.comp_graph.valvar import ValVar
from mizore.operators import QubitOperator
from mizore.operators.observable import Observable
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.error_mitigation.error_extrapolation import ErrorExtrapolation
from mizore.transpiler.measurement.infinite import InfiniteMeasurement
from mizore.transpiler.measurement.naive import NaiveMeasurement
from mizore.transpiler.noise_model.simple_noise import DepolarizingNoise
from mizore.transpiler.param_circuit.gradient import GradientCircuit
from mizore import jax_array

def simple_pqc_node_one_obs(param_var = 0.001, name=None):
    n_qubit = 4
    hamil = simple_4_qubit_lih()
    ops = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    ops += QubitOperator('Z0') + QubitOperator('Z1') + QubitOperator('Z2') + QubitOperator('Z3')
    ops += QubitOperator('X0 X1 X2 X3')
    ops2 = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    #ops += QubitOperator('X1 X2')
    #ops += QubitOperator('X2 X3')
    obs = Observable(n_qubit, hamil)
    bc = MetaCircuit(n_qubit, [GateGroup(X(0)), RotationGroup(ops),
                               RotationGroup(ops2, fixed_angle_shift=[0.2])])
    n_param = bc.n_param
    node = MetaCircuitNode(bc, obs, name=name)
    node.shot_num.set_value(1000)
    params = ValVar([0.0]*n_param, [param_var]*n_param)
    node.params.set_value(params)
    return node

node_valvars = []
grad_valvars = []
param_valvars = []

lr = 0.3
n_step = 5
init_shot_num = 100000
node = simple_pqc_node_one_obs(param_var=0.000, name="Init")
node_valvars.append(node())
params = node.params.replica()
grads_dict = GradientCircuit(init_shot_num=init_shot_num) | node
grad = grads_dict[node]
cg = CompGraph([grad, node()])
grad_valvars.append(grad)

for i in range(n_step):
    params = params - grad*lr
    node = simple_pqc_node_one_obs(param_var=0.000, name=f"Step{i+1}")
    node_valvars.append(node())
    node.params.set_value(params)
    param_valvars.append(node.params)
    grads_dict = GradientCircuit(init_shot_num=init_shot_num) | node
    grad = grads_dict[node]
    grad_valvars.append(grad)

params = params - grad*lr
node = simple_pqc_node_one_obs(param_var=0.000, name="Output")
node_valvars.append(node())
node.params.set_value(params)
param_valvars.append(node.params)

cg = CompGraph(node_valvars)

for node in cg.all().by_type(QCircuitNode):
    node.has_random = True
    node.random_config = {"use_dm": True}


SimpleReducer() | cg
DepolarizingNoise(error_rate=0.001) | cg
ErrorExtrapolation([1.1, 1.2]) | cg

#print(cg)

#print(cg)

i = 0
for layer in cg.layers():
    print("layer",i)
    i+=1
    #NaiveMeasurement(state_ignorant=True) | layer
    InfiniteMeasurement() | layer
    CircuitRunner(n_proc=4, cache_key=1) | layer

eval_fun, var_list, init_list = node().val.get_eval_fun()

print("val")

for inner_exp in node_valvars:
    inner_exp.mean.show_value()
print("var")
for inner_exp in node_valvars:
    inner_exp.var.show_value()