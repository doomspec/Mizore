from .block import Block


class Gates(Block):
    def __init__(self, *gates):
        Block.__init__(self, 0)
        self.gates = list(gates)

    def get_gates(self, params):
        return self.gates


