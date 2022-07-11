from mizore.backend_circuit.one_qubit_gates import Hadamard, X
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator


def test_expv():
    n_qubit = 1
    blocks = [Gates(Hadamard(0))]
    ref_circuit = MetaCircuit(n_qubit, blocks)
    expv = ref_circuit.get_expectation_value(QubitOperator("X0"))
    assert abs(expv - 1.0) < 1e-11