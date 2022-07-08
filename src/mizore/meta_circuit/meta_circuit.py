from __future__ import annotations

from copy import copy
from typing import List, Tuple, Union

from mizore.meta_circuit.block.gate_group import GateGroup
from mizore.backend_circuit.gate import Gate
from mizore.meta_circuit.block.block import Block
from mizore.backend_circuit.backend_circuit import BackendCircuit, BackendState, BackendOperator
from mizore import np_array
from mizore.operators import QubitOperator


class MetaCircuit:
    def __init__(self, n_qubit: int, blocks: List[Block] = None, gates: List[Gate] = None):
        self.n_qubit: int = n_qubit
        self._blocks: List[Block]
        if blocks is not None:
            self._blocks = copy(blocks)
            assert gates is None
        elif gates is not None:
            self._blocks = [GateGroup(*gates)]
        else:
            self._blocks = []
        self._n_param = -1
        self.param_delimiter = None
        self.post_processors = []
        self.has_random = False
        self.default_params = None

    def replica(self) -> MetaCircuit:
        res = copy(self)
        res._blocks = copy(self._blocks)
        res.post_processors = copy(self.post_processors)
        return res

    def make_param_delimiter(self):
        param_delimiter = [0] * (len(self._blocks) + 1)
        for i in range(len(self._blocks)):
            param_delimiter[i + 1] = param_delimiter[i] + self._blocks[i].n_param
        self.param_delimiter = param_delimiter

    def add_gates(self, gates: List[Gate]):
        self.add_blocks([GateGroup(*gates)])

    @property
    def blocks(self):
        """
        :return: a copy of the list of blocks
        """
        return self._blocks[:]

    def add_blocks(self, blocks: List[Block]):
        self._blocks.extend(blocks)
        self._n_param = -1
        self.param_delimiter = None

    def set_blocks(self, blocks: List[Block]):
        self._blocks = blocks
        self._n_param = -1
        self.param_delimiter = None

    @property
    def n_param(self):
        if self._n_param != -1:
            return self._n_param
        _n_para = 0
        for block in self._blocks:
            _n_para += block.n_param
        self._n_param = _n_para
        return _n_para

    def add_post_process(self, processor):
        self.post_processors.append(processor)

    def post_process(self, gates_list, block_list) -> List[Gate]:
        new_gates_list = gates_list
        for processor in self.post_processors:
            new_gates_list = processor(new_gates_list, block_list)
        # Concatenate gate list of each block
        gate_list = []
        for gates in new_gates_list:
            gate_list.extend(gates)
        return gate_list

    def get_gates(self, params=None):
        if self.param_delimiter is None:
            self.make_param_delimiter()
        if isinstance(params, List):
            params = np_array(params, copy=False)
        if params is None:
            params = np_array([0.0] * self.n_param) if self.default_params is None else self.default_params
        blocks = self._blocks
        delimiter = self.param_delimiter
        origin_gates_list = []
        for i in range(len(blocks)):
            origin_gates_list.append(blocks[i].get_gates(params[delimiter[i]:delimiter[i + 1]]))
        gate_list = self.post_process(origin_gates_list, blocks)
        return gate_list

    def get_backend_circuit(self, params=None) -> BackendCircuit:
        backend_circuit = BackendCircuit(self.n_qubit, self.get_gates(params))
        return backend_circuit

    def get_backend_state(self, params=None, dm=False):
        backend_circuit = self.get_backend_circuit(params)
        return backend_circuit.get_quantum_state(dm=dm)

    def get_expectation_value(self, op: Union[QubitOperator, BackendOperator], params=None, use_dm=False):
        backend_circuit = self.get_backend_circuit(params)
        backend_state = BackendState(self.n_qubit, dm=use_dm)
        backend_circuit.update_quantum_state(backend_state)
        if isinstance(op, BackendOperator):
            return op.get_expectation_value(backend_state)
        else:
            backend_op = BackendOperator(op)
            return backend_op.get_expectation_value(backend_state)

    def get_block_index_by_param_index(self, param_index) -> Tuple[int, int]:
        if self.param_delimiter is None:
            self.make_param_delimiter()
        # Figure out which block this param is in
        block_index = 0
        for delimit in self.param_delimiter:
            if delimit > param_index:
                break
            block_index += 1
        block_index -= 1
        in_block_param_index = param_index - self.param_delimiter[block_index]
        return block_index, in_block_param_index

    def get_gradient_circuits(self, param_index) -> List[Tuple[float, MetaCircuit]]:
        block_index, in_block_param_index = self.get_block_index_by_param_index(param_index)
        gradient_blocks = self._blocks[block_index].get_gradient_blocks(in_block_param_index)
        new_circuits = []
        for coeff, block in gradient_blocks:
            new_blocks = self._blocks[:block_index]
            new_blocks.append(block)
            new_blocks.extend(self._blocks[block_index + 1:])
            new_circuits.append((coeff, MetaCircuit(self.n_qubit, new_blocks)))
        return new_circuits

    def get_zero_param(self):
        return np_array([0.0] * self.n_param)

    def get_fixed_param_circuit(self, params=None):
        return MetaCircuit(self.n_qubit, gates=self.get_gates(params))

    def __str__(self):
        gate_list = self.get_gates()
        return "\n".join(map(lambda gate: str(gate), gate_list))
