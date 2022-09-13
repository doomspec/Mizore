from chemistry.simple_mols import simple_4_qubit_lih
from mizore.comp_graph.value import Variable
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.backend_circuit.one_qubit_gates import X
from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_optimize.simple_reducer import SimpleReducer
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.error_mitigation.error_extrapolation import ErrorExtrapolation
from mizore.transpiler.measurement.infinite import InfiniteMeasurement
from mizore.transpiler.measurement.l1 import L1Sampling
from mizore.transpiler.noise_model.simple_noise import DepolarizingNoise
from mizore.transpiler.param_circuit.gradient import GradientCircuit
from mizore import np_array
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal


def simple_pqc_node_one_obs(param_var=0.001, name=None):
    n_qubit = 4
    hamil = simple_4_qubit_lih()
    ops = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    ops += QubitOperator('Z0') + QubitOperator('Z1') + QubitOperator('Z2') + QubitOperator('Z3')
    ops += QubitOperator('X0 X1 X2 X3')
    ops2 = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    obs = hamil
    bc = MetaCircuit(n_qubit, [Gates(X(0)), RotationGroup(ops),
                               RotationGroup(ops2, fixed_angle_shift=[0.2])])
    n_param = bc.n_param
    node_ = DeviceCircuitNode(bc, obs, name=name)
    node_.shot_num.set_value(1000)
    params_ = Variable([0.0] * n_param, [param_var] * n_param)
    node_.params.bind_to(params_)
    return node_


def read_ans(node_expvs_):
    mean_list = []
    for inner_exp in node_expvs_:
        mean_list.append(inner_exp.value())
    var_list = []
    for inner_exp in node_expvs_:
        var_list.append(inner_exp.var.value())
    return np_array(mean_list), np_array(var_list)


very_small = 1e-7
n_proc = 4

node_expvs = []

lr = 0.3
n_step = 5
init_shot_num = 1000
node = simple_pqc_node_one_obs(param_var=0.000, name="Init")
node_expvs.append(node())
params = node.params.replica()
grads_dict = GradientCircuit(init_shot_num=init_shot_num) | node
grad = grads_dict[node]
cg = CompGraph([grad, node()])

for i in range(n_step):
    params = params - grad * lr
    node = simple_pqc_node_one_obs(param_var=0.000, name=f"Step{i + 1}")
    node_expvs.append(node())
    node.params.bind_to(params)
    grads_dict = GradientCircuit(init_shot_num=init_shot_num) | node
    grad = grads_dict[node]

params = params - grad * lr
node = simple_pqc_node_one_obs(param_var=0.000, name="Output")
node_expvs.append(node())
node.params.bind_to(params)

cg = CompGraph(node_expvs)

test_vanilla = False
if test_vanilla:
    for layer in cg.layers():
        InfiniteMeasurement() | layer
        CircuitRunner(n_proc=n_proc) | layer
    means, vars = read_ans(node_expvs)
    assert norm(means - np_array(
        [0.01584071, 0.00974723, 0.00657693, 0.00484444, 0.00382561, 0.00317085, 0.00271108])) < very_small
    cg.del_all_cache()

SimpleReducer() | cg
DepolarizingNoise(error_rate=0.001) | cg
for node in cg.all().by_type(QCircuitNode):
    node.config = {"use_dm": True}

test_noisy = False
if test_noisy:
    for layer in cg.layers():
        InfiniteMeasurement() | layer
        CircuitRunner(n_proc=n_proc) | layer
    means, vars = read_ans(node_expvs)
    assert norm(means - np_array(
        [0.02823547, 0.022537582, 0.019505372, 0.01782053, 0.016821358, 0.01617906, 0.015730202])) < very_small
    cg.del_all_cache()


def test_finite_measurement():
    for layer in cg.layers():
        L1Sampling() | layer
        CircuitRunner(n_proc=n_proc) | layer
    means, vars = read_ans(node_expvs)
    assert_array_almost_equal(means , [0.02823547, 0.02254955, 0.019525697, 0.01786764,
                                       0.016880326, 0.016203582, 0.015850462])
    assert_array_almost_equal(vars, [3.935065e-05, 3.935065e-05, 3.935065e-05, 3.935065e-05,
                                     3.935065e-05, 3.935065e-05, 3.935065e-05])
    cg.del_all_cache()


def test_error_extrap():
    ErrorExtrapolation([1.1, 1.2]) | cg
    for layer in cg.layers():
        InfiniteMeasurement() | layer
        CircuitRunner(n_proc=n_proc) | layer
    means, vars = read_ans(node_expvs)
    assert_array_almost_equal(means, [0.01584542, 0.00975204, 0.0065819, 0.00484943, 0.00383091, 0.00317609, 0.00271624])
    cg.del_all_cache()
