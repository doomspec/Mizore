from .block import Block
from mizore import np_array

class FixedBlock(Block):
    def __init__(self, block: Block, params = None):
        Block.__init__(self, 0)
        if params is None:
            params_ = np_array([0.0]*block.n_param)
        else:
            params_ = np_array(params)
        self.gates = list(block.get_gates(params_))

    def get_gates(self, params):
        return self.gates


