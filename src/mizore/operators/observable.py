from qulacs import GeneralQuantumOperator

from .qubit_operator import QubitOperator


class Observable:

    def __init__(self, n_qubit, operator: QubitOperator):
        """
        :param n_qubit: Number of qubits
        :param operator: The qubit operator
        :param precision: The required precision
        """
        self._operator: QubitOperator = operator
        self.n_qubit = n_qubit
        self.qulacs_operator = None

    @property
    def operator(self):
        return self._operator

    def get_backend_operator(self) -> GeneralQuantumOperator:
        return Observable.make_qulacs_operator(self)

    @classmethod
    def make_qulacs_operator(cls, obs) -> GeneralQuantumOperator:
        if obs.qulacs_operator is not None:
            return obs.qulacs_operator
        operator = GeneralQuantumOperator(obs.n_qubit)
        for qset, pauli_ops, weight in obs._operator.qset_ops_weight():
            operator.add_operator(weight, pauliword_to_qulacs_string(qset, pauli_ops))
        obs.qulacs_operator = operator
        return operator


index_to_pauli_name = {
    0: "I",
    1: "X",
    2: "Y",
    3: "Z"
}


def pauliword_to_qulacs_string(qset, pauli_ops):
    return " ".join(map(lambda index, pauli: "{} {}".format(index_to_pauli_name[pauli], str(index)), qset, pauli_ops))
