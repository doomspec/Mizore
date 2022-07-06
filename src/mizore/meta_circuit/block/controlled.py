from __future__ import annotations

from typing import Tuple, Union, List

from .block import Block

from ...backend_circuit.controlled import ControlledGate


class Controlled(Block):
    def __init__(self, controlled_block: Block, control_qset, trigger_values:Union[List, None]=None):
        Block.__init__(self, controlled_block.n_param)
        self.controlled_block = controlled_block
        self.control_qset = control_qset
        self.trigger_values: List[int, ...]
        if trigger_values is None:
            self.trigger_values = [1]*len(self.control_qset)
        else:
            self.trigger_values = trigger_values
        assert len(control_qset) == 1

    def get_gates(self, params):
        gates_to_control = self.controlled_block.get_gates(params)
        controlled_gates = [None]*len(gates_to_control)
        for i in range(len(gates_to_control)):
            controlled_gates[i] = ControlledGate(gates_to_control[i], self.control_qset[0], self.trigger_values[0])
        return controlled_gates

    def get_inverse_block(self):
        return Controlled(self.controlled_block.get_inverse_block(), self.control_qset)

    def get_gradient_blocks(self, param_index: int, params=None):
        pass