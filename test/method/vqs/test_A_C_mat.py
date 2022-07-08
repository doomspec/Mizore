from chemistry.simple_mols import simple_4_qubit_lih
from method.vqs.finite_diff_A_C_mat import get_A_by_finite_diff, get_C_by_finite_diff
from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block.fixed_block import FixedBlock
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.inner_product_circuits import A_mat_real, C_mat_imag, C_mat_real
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from numpy.testing import assert_array_almost_equal, assert_allclose


def test_A_real_C_imag():
    hamil = simple_4_qubit_lih()

    blocks = []

    for qset_op_weight in hamil.qset_op_weight():
        if len(qset_op_weight[0]) == 0:
            continue
        blocks.append(
            FixedBlock(Rotation(qset_op_weight[0], qset_op_weight[1], qset_op_weight[2], angle_shift=0.6)))

    i = 0
    for qset_op_weight in hamil.qset_op_weight():
        if len(qset_op_weight[0]) == 0:
            continue
        blocks.append(Rotation(qset_op_weight[0], qset_op_weight[1], qset_op_weight[2], angle_shift=0.0))
        if i > 8:
            break
        i += 1

    circuit = MetaCircuit(4, blocks=blocks)

    param = Value([0.5]*circuit.n_param)

    A = A_mat_real(circuit, param)
    C_imag = C_mat_imag(circuit, hamil, param)
    C_real = C_mat_real(circuit, hamil, param)

    cg = CompGraph([A, C_imag, C_real])

    node: DeviceCircuitNode
    for node in cg.by_type(DeviceCircuitNode):
        node.shot_num.set_value(1e11)
        node.expv_shift_from_var = False

    CircuitRunner() | cg

    A_analytical = A.value()
    C_imag_analytical = C_imag.value()
    C_real_analytical = C_real.value()

    A_finite = get_A_by_finite_diff(circuit, param.value()).real
    C_finite = get_C_by_finite_diff(circuit, hamil, param.value())

    assert_array_almost_equal(A_finite, A_analytical, decimal=5)
    assert_allclose(C_finite.imag, C_imag_analytical, rtol=0.01, atol=0.0001)
    assert_allclose(C_finite.real, C_real_analytical, rtol=0.01, atol=0.0001)
