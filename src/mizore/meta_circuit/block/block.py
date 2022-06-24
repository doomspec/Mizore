from __future__ import annotations

from typing import List

from mizore.backend_circuit.gate import Gate

from jax.numpy import array

class Block:

    def __init__(self, n_param: int, fixed_param:List = None):
        self.n_param: int = n_param
        self._fixed_param = fixed_param if fixed_param is not None else array([0.0]*n_param)
        # A dict for transpilers to add tags for processing
        # For example tag = {"noisy":True} can remind certain transpiler to add noise
        self.attr = {}

    @property
    def fixed_param(self):
        return self._fixed_param

    def get_gradient_blocks(self, param_index: int) -> List[Gate]:
        raise NotImplementedError()

    def get_gates(self, params: List) -> List[Gate]:
        raise NotImplementedError()

    def get_inverse_block(self) -> Block:
        raise NotImplementedError()
