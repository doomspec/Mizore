from chemistry.simple_mols import simple_4_qubit_lih
from method.vqs.finite_diff_A_C_mat import get_A_by_finite_diff, get_C_by_finite_diff
from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.meta_circuit.block.fixed_block import FixedBlock
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.inner_product_circuits import A_mat_real, C_mat_imag
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from numpy.testing import assert_array_almost_equal

def test_A_real_C_imag():

    hamil = simple_4_qubit_lih()

    blocks = []

    for qset_ops_weight in hamil.qset_ops_weight():
        if len(qset_ops_weight[0]) == 0:
            continue
        blocks.append(
            FixedBlock(Rotation(qset_ops_weight[0], qset_ops_weight[1], qset_ops_weight[2], fixed_angle_shift=0.6)))

    i = 0
    for qset_ops_weight in hamil.qset_ops_weight():
        if len(qset_ops_weight[0]) == 0:
            continue
        blocks.append(Rotation(qset_ops_weight[0], qset_ops_weight[1], qset_ops_weight[2], fixed_angle_shift=0.0))
        if i > 8:
            break
        i += 1

    circuit = MetaCircuit(4, blocks=blocks)

    A = A_mat_real(circuit)
    C = C_mat_imag(circuit, hamil)

    cg = CompGraph([A, C])

    for node in cg.by_type(DeviceCircuitNode):
        node.shot_num.set_value(1e11)

    CircuitRunner() | cg

    A_analytical = A.value()
    C_analytical = C.value()
    A_finite = get_A_by_finite_diff(circuit).real
    C_finite = get_C_by_finite_diff(circuit, hamil).imag

    assert_array_almost_equal(A_finite, A_analytical, decimal=5)
    assert_array_almost_equal(C_finite, C_analytical, decimal=5)