from typing import List

from .block import Block
from ...backend_circuit.gate import Gate


class Gates(Block):
    def __init__(self, *gates):
        Block.__init__(self, 0)
        self.gates = list(gates)

    @classmethod
    def from_gate_list(cls, gate_list: List[Gate]):
        """
        Create a Gates object by the given gate_list. The gate_list provided will not be copied.
        :param gate_list:
        :return:
        """
        block = Gates()
        block.gates = gate_list
        return block

    def get_gates(self, params):
        return self.gates


