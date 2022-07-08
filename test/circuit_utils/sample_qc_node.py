from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit



def simple_qc_node():
    n_qubit = 3
    ops = QubitOperator('X0 Z1 X2') + QubitOperator('X0 Y1 Z2')
    obs1 = QubitOperator('Z0')+QubitOperator('X0')+QubitOperator('Y0')
    #obs2 = Observable(n_qubit, QubitOperator('Z1')+QubitOperator('X1')+QubitOperator('Y1'))
    blk = RotationGroup(ops, fixed_angle_shift=[1.0, 1.0])
    bc = MetaCircuit(n_qubit, [blk, blk])
    node = DeviceCircuitNode(bc.get_fixed_param_circuit(), obs1)
    return node