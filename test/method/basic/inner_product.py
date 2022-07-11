from mizore.comp_graph.comp_graph import CompGraph
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.basic.inner_product import get_inner_prod_by_additional
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner


def test_get_inner_prod_by_additional():
    n_qubit = 3
    blocks = [Rotation((i,), (1,), angle_shift=1.0) for i in range(n_qubit)]
    blocks.append(Rotation((0, 1, 2), (1, 1, 3), angle_shift=1.5))
    ref_circuit = MetaCircuit(n_qubit, blocks)
    additional_blocks = [Rotation((0, 1, 2), (2, 2, 1), angle_shift=-1.5)]

    state1 = ref_circuit.get_backend_state()
    circuit_with_addition = MetaCircuit(n_qubit, blocks + additional_blocks)
    state2 = circuit_with_addition.get_backend_state()
    innerp_expected = state1.inner_product(state2)

    innerp = get_inner_prod_by_additional(ref_circuit, additional_blocks)
    cg = CompGraph(innerp)
    CircuitRunner() | cg

    print(innerp_expected, innerp.value())

    assert abs(innerp.value() - innerp_expected) < 1e-8
