from __future__ import annotations

from .block import Block

from ...backend_circuit.controlled import ControlledGate


class Controlled(Block):
    def __init__(self, controlled_block: Block, control_qset):
        Block.__init__(self, controlled_block.n_param)
        self.controlled_block = controlled_block
        self.control_qset = control_qset
        assert len(control_qset) == 1

    def get_gates(self, params):
        controlled_gates = self.controlled_block.get_gates(params)
        for i in range(len(controlled_gates)):
            controlled_gates[i] = ControlledGate(controlled_gates[i], self.control_qset[0])
        return controlled_gates

    def get_inverse_block(self):
        return Controlled(self.controlled_block.get_inverse_block(), self.control_qset)

    def get_gradient_blocks(self, param_index: int, params=None):
        pass