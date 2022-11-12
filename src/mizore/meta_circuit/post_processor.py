from typing import List

from mizore.meta_circuit.block import Block
from mizore.backend_circuit.gate import Gate


class SimpleProcessor:

    def __call__(self, gates_list: List[List[Gate]], blocks: List[Block]):
        new_gates_list = []
        for i in range(len(gates_list)):
            new_gates_list.append(self.process(gates_list[i], blocks[i]))
        return new_gates_list

    def process(self, gates: List[Gate], block: Block):
        raise NotImplementedError()
