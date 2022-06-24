from typing import List

from mizore.meta_circuit.block.block import Block


class swap_router(Block):
    def __init__(self, qubit_route: List[int]):
        super().__init__(0)
        self.qubit_route = qubit_route
