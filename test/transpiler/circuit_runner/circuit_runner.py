from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Variable
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from mizore.transpiler.measurement.l1sampling import L1Sampling
from mizore import np_array


def simple_pqc_node(param_var = 0.001):
    n_qubit = 2
    ops = QubitOperator('X0 Z1') + QubitOperator('X0 Y1')
    obs1 = QubitOperator('Z0')+QubitOperator('X0')
    obs2 = QubitOperator('Z1')+QubitOperator('X1')
    blk = RotationGroup(ops, fixed_angle_shift=np_array([2.0, 1.0]))
    bc = MetaCircuit(n_qubit, [blk, blk])
    node = DeviceCircuitNode(bc, [obs1, obs2], param=Variable(np_array([0.0]*4), np_array([param_var]*4)))
    return node

#node0 = simple_pqc_node(param_var=0.0)
node1 = simple_pqc_node(param_var=0.001)
#exp_valvar0 = node0()
expv = node1()
cg = CompGraph([expv])
CircuitRunner() | cg
L1Sampling() | cg

#node1.values.show_value()

expv.show_value()
expv.var.show_value()
