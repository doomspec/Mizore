from .block import Block


class GateGroup(Block):
    def __init__(self, gates):
        Block.__init__(self, 0)
        self.gates = gates

    def get_gates(self, params):
        return self.gates
