from typing import Tuple, List

from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.quantum_circuit import QuantumCircuit


def count_one_two_qubit_gates(gates: List[Gate]) -> Tuple[int, int]:
    n_one = 0
    n_two = 0
    for gate in gates:
        if len(gate.qset) == 1:
            n_one += 1
        elif len(gate.qset) == 2:
            n_two += 1
        else:
            raise Exception("backend_circuit contains more than one/two qubit gates. {} received.".format(gate))
    return n_one, n_two
