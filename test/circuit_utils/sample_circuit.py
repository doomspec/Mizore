from mizore.backend_circuit.one_qubit_gates import X
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.block.rotation_group import RotationGroup
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator


def circuit_for_test_0():
    ops = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    ops += QubitOperator('Z0') + QubitOperator('Z1') + QubitOperator('Z2') + QubitOperator('Z3')
    ops += QubitOperator('X0 X1 X2 X3')
    ops2 = QubitOperator('X0') + QubitOperator('X1') + QubitOperator('X2') + QubitOperator('X3')
    circuit = MetaCircuit(4, [Gates(X(0)), RotationGroup(ops),
                               Rotation((0, 1, 2, 3), (1, 1, 1, 1), 0.2, angle_shift=2.0),
                               RotationGroup(ops2, fixed_angle_shift=[0.5])])
    return circuit