from __future__ import annotations

from typing import List

from mizore.backend_circuit.gate import Gate
from mizore.backend_circuit.rotations import SingleRotation
from mizore.backend_circuit.utils import merge_qset



from qulacs.gate import to_matrix_gate

def default_reducer(ctrl: ControlledGate):
    gates = ctrl.controlled_gate.simple_reduce()
    if gates is None:
        return None
    for i in range(len(gates)):
        gates[i] = ControlledGate(gates[i], ctrl.control_index, ctrl.trigger_value)
    return gates

special_reducer = {}

class ControlledGate(Gate):
    def __init__(self, controlled_gate: Gate, control_index: int, trigger_value=1):
        self.controlled_gate = controlled_gate
        self.control_index = control_index
        self.trigger_value = trigger_value
        super().__init__(merge_qset(controlled_gate.qset, (control_index,)))

    @property
    def qulacs_gate(self):
        qulacs_gate = to_matrix_gate(self.controlled_gate.qulacs_gate)
        qulacs_gate.add_control_qubit(self.control_index, self.trigger_value)
        return qulacs_gate

    def simple_reduce(self) -> List[Gate]:
        """
        This is a tricky function, where a lot of optimization can be done
        :return:
        """
        reducer = special_reducer.get(self.controlled_gate.__class__.__name__, None)
        if reducer is None:
            reducer = default_reducer
        return reducer(self)


def PauliRotation_reducer(ctrl: ControlledGate):
    gates = ctrl.controlled_gate.simple_reduce()
    middle_index = len(gates) // 2
    gates[middle_index] = ControlledGate(gates[middle_index], ctrl.control_index, ctrl.trigger_value)
    return gates


special_reducer["PauliRotation"] = PauliRotation_reducer

def GlobalPhase_reducer(ctrl: ControlledGate):
    gates = [SingleRotation(3, ctrl.control_index, ctrl.controlled_gate.angle*2)]
    return gates

special_reducer["GlobalPhase"] = GlobalPhase_reducer

