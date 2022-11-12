from typing import Tuple

from qulacs import GeneralQuantumOperator

from mizore.operators import QubitOperator


def paulistring_to_qulacs_string(paulistring):
    return " ".join(
        map(lambda pauli: "{} {}".format(pauli[1], pauli[0]), paulistring))


def iter_qulacs_ops(ops: QubitOperator, n_qubit) -> Tuple[GeneralQuantumOperator, float]:
    for op, weight, in ops.terms.items():
        if len(op) == 0:
            continue
        qulacs_op = GeneralQuantumOperator(n_qubit)
        qulacs_op.add_operator(1.0, paulistring_to_qulacs_string(op))
        yield qulacs_op, weight


index_to_pauli_name = {
    0: "I",
    1: "X",
    2: "Y",
    3: "Z"
}
