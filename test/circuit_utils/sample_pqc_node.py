from chemistry.simple_mols import simple_4_qubit_lih
from mizore.backend_circuit.one_qubit_gates import X
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Variable
from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.meta_circuit.block.rotation import Rotation
from mizore.operators import QubitOperator
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators.observable import Observable
from mizore import np_array


def simple_pqc_node_one_obs(param_var=0.001, name=None):
    n_qubit = 2
    ops = QubitOperator('X0 Z1') + QubitOperator('X0 Y1')
    obs1 = Observable(n_qubit, QubitOperator('Z0') + QubitOperator('X0') + QubitOperator('Y0'))
    blk = RotationGroup(ops, fixed_angle_shift=[2.0, 1.0])
    bc = MetaCircuit(n_qubit, [blk, blk])
    node = DeviceCircuitNode(bc, [obs1], name=name)
    node.shot_num.set_value(1000)
    params = Variable([0.0] * 4, [param_var] * 4)
    node.params.bind_to(params)
    return node


def simple_pqc_node(param_var=0.001):
    n_qubit = 2
    ops = QubitOperator('X0 Z1') + QubitOperator('X0 Y1')
    obs1 = Observable(n_qubit, QubitOperator('Z0') + QubitOperator('X0'))
    obs2 = Observable(n_qubit, QubitOperator('Z1') + QubitOperator('X1'))
    blk = RotationGroup(ops, fixed_angle_shift=np_array([2.0, 1.0]))
    bc = MetaCircuit(n_qubit, [blk, blk])
    node = DeviceCircuitNode(bc, [obs1, obs2])
    params = Variable([0.0] * 4, [param_var] * 4)
    node.params.bind_to(params)
    return node


def simple_large_pqc_node(param_var=0.001):
    n_qubit = 4
    hamil = simple_4_qubit_lih()
    ops = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    ops += QubitOperator('Z0') + QubitOperator('Z1') + QubitOperator('Z2') + QubitOperator('Z3')
    ops += QubitOperator('X0 X1 X2 X3')
    ops2 = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    obs = Observable(n_qubit, hamil)
    bc = MetaCircuit(n_qubit, [GateGroup(X(0)), RotationGroup(ops),
                               RotationGroup(ops2, fixed_angle_shift=[1.0])])
    n_param = bc.n_param
    node = DeviceCircuitNode(bc, obs)
    node.shot_num.set_value(1000)
    params = Variable([0.0] * n_param, [param_var] * n_param)
    node.params.bind_to(params)
    return node


def simple_large_pqc_node2(param_var=0.001):
    n_qubit = 4
    hamil = simple_4_qubit_lih()
    ops = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    ops += QubitOperator('Z0') + QubitOperator('Z1') + QubitOperator('Z2') + QubitOperator('Z3')
    ops += QubitOperator('X0 X1 X2 X3')
    ops2 = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    obs = Observable(n_qubit, hamil)
    obs2 = Observable(n_qubit, 2 * QubitOperator('X0'))
    bc = MetaCircuit(n_qubit, [GateGroup(X(0)), RotationGroup(ops),
                               Rotation((0, 1, 2, 3), (1, 1, 1, 1), 0.2, fixed_angle_shift=2.0),
                               RotationGroup(ops2, fixed_angle_shift=[0.5])])
    n_param = bc.n_param
    node = DeviceCircuitNode(bc, [obs, obs2])
    node.shot_num.set_value(1000)
    params = Variable([0.0] * n_param, [param_var] * n_param)
    node.params.bind_to(params)
    return node
