from copy import copy
from typing import Tuple, List

from mizore.backend_circuit.one_qubit_gates import Hadamard
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block import Block
from mizore.meta_circuit.block.controlled import Controlled
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode


def get_inner_prod_by_additional(ref_circuit: MetaCircuit, additional_blocks: List[Block]) -> Value:
    n_qubit = ref_circuit.n_qubit + 1

    new_circuit = ref_circuit.replica()
    new_circuit.n_qubit += 1

    new_blocks = [Gates(Hadamard(n_qubit - 1))]
    for block in additional_blocks:
        new_blocks.append(Controlled(block, (n_qubit - 1,)))

    new_circuit.add_blocks(new_blocks)

    node = DeviceCircuitNode(new_circuit, [QubitOperator(f"X{n_qubit - 1}"), QubitOperator(f"Y{n_qubit - 1}")],
                             "InnerProd")

    innerp = Value.unary_operator(node(), lambda x: x[0] + 1.0j * x[1])

    return innerp
