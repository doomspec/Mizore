from __future__ import annotations

from copy import copy
from typing import List

from mizore.meta_circuit.block.block import Block


class Combined(Block):

    def __init__(self, blocks: List[Block]):
        param_delimiter = [0]*(len(blocks)+1)
        for i in range(len(blocks)):
            param_delimiter[i+1] = param_delimiter[i]+blocks[i].n_param
        self.param_delimiter = param_delimiter
        self.inner_blocks = copy(blocks)
        Block.__init__(self, param_delimiter[-1])

    def get_gates(self, params):
        gates = []
        for i in range(len(self.inner_blocks)):
            gates.extend(self.inner_blocks[i].get_gates(params[self.param_delimiter[i]:self.param_delimiter[i + 1]]))
        return gates

    def get_derivative_gates(self, params, param_index):
        gates = []
        for i in range(len(self.inner_blocks)):
            gates.extend(self.inner_blocks[i].get_derivative_gates(params[self.param_delimiter[i]:self.param_delimiter[i + 1]]))
        return gates

    def get_inverse_block(self) -> Combined:
        n_blocks = len(self.inner_blocks)
        new_blocks: List[Block] = [None]*n_blocks
        for i in range(n_blocks):
            new_blocks[n_blocks-i-1] = self.inner_blocks[i].get_inverse_block()
        return Combined(new_blocks)

