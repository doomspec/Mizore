from mizore.meta_circuit.block.controlled import Controlled
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.trotter import get_trotter, Trotter
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.transpiler.replace.replacer import Replacer


class Trotterizer(Replacer):
    def __init__(self, max_delta_t, method="first-order"):
        super().__init__()
        self.method = method
        self.max_delta_t = max_delta_t

    def replace(self, circuit: MetaCircuit):
        new_blocks = []
        for block in circuit.blocks:
            if isinstance(block, Controlled):
                block.controlled_block = self.new_block(block.controlled_block)
                new_blocks.append(block)
                continue
            new_blocks.append(self.new_block(block))
        circuit.set_blocks(new_blocks)

    def new_block(self, block):
        if not isinstance(block, ExactEvolution) or (not block.to_decompose):
            return block
        else:
            return Trotter(block.hamil, self.max_delta_t, init_time=block.fixed_param[0])
