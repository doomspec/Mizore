from qulacs import QuantumCircuit as qulacsQCircuit


class QuantumCircuit():
    def __init__(self, n_qubit):
        self.gates = []
        self.n_qubit = n_qubit

    def add_gate(self, gate):
        self.gates.append(gate)

    def add_gates(self, gates):
        self.gates.extend(gates)

    def get_qulacs_circuit(self) -> qulacsQCircuit:
        qulacs_circuit = qulacsQCircuit(self.n_qubit)
        for gate in self.gates:
            qulacs_circuit.add_gate(gate.qulacs_gate)
        return qulacs_circuit

    def __iter__(self):
        return self.gates.__iter__()

    def __str__(self):
        res = map(lambda gate: str(gate), self.gates)
        return "\n".join(res)
