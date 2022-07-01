from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore import np_array
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.operators.observable import Observable

n_qubit = 2
ops = QubitOperator('Z0 Z1') + QubitOperator('X0 Y1')

obs = Observable(n_qubit, QubitOperator('Z0'))
blk = RotationGroup(ops)

bc = MetaCircuit(n_qubit, [blk, blk])
param = bc.get_zero_param()

param += np_array([1.0, 1.0, 1.0, 1.0])

eps = 0.001
y0 = bc.get_expectation_value(obs, param)
param += np_array([0.0, eps, 0.0, 0.0])
y1 = bc.get_expectation_value(obs, param)
print((y1-y0)/eps)
gradient_bc = bc.get_gradient_circuits(1)
g=0
for coeff, gbc in gradient_bc:
    g+=coeff*gbc.get_expectation_value(obs, param)
print(g)