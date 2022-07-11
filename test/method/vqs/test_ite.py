from mizore.comp_graph.comp_graph import CompGraph
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block.rotation import Rotation
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.imag_time_evol import imag_evol_gradient
from mizore.operators import QubitOperator
from mizore.operators.spectrum import get_first_k_eigenstates
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from numpy.testing import assert_allclose
from numpy import exp, sqrt


def verify_ite_for_single_qubit(energy_list, hamil, step_size):
    eigenvalue, eigenstate = get_first_k_eigenstates(2, 1, hamil, sparse=False)
    Eg, Ee = eigenvalue
    E0 = energy_list[0]
    a0 = sqrt((E0 - Ee) / (Eg - Ee))
    a1 = sqrt(1 - a0 ** 2)
    expect_energy_list = []

    for i in range(len(energy_list)):
        expect_energy_list.append((a0 ** 2) * Eg + (a1 ** 2) * Ee)
        a0 = a0 * exp(-Eg * step_size)
        a1 = a1 * exp(-Ee * step_size)
        C = sqrt(a0 ** 2 + a1 ** 2)
        a0 /= C
        a1 /= C

    assert_allclose(energy_list, expect_energy_list, rtol=0.02, atol=0.01)


def test_single_qubit():
    n_qubit = 1
    step_size = 5e-3

    hamil = QubitOperator("Z0") - QubitOperator("X0") + QubitOperator("Y0") + 2.0

    blocks = [Rotation((0,), (1,), angle_shift=1.0),
              Rotation((0,), (3,), angle_shift=1.5),
              Rotation((0,), (1,), angle_shift=1.0)]

    circuit = MetaCircuit(n_qubit, blocks)

    curr_time = 0.0
    param = Value([0.0] * circuit.n_param)

    energy_list = []
    for i in range(50):
        energy = DeviceCircuitNode(circuit, hamil, param=param)()
        energy_list.append(energy)
        evol_grad, A_real, C_real = imag_evol_gradient(circuit, hamil, param)
        param = param - evol_grad * step_size
        curr_time += step_size
    energy = DeviceCircuitNode(circuit, hamil, param=param)()
    energy_list.append(energy)

    cg = CompGraph(energy_list)

    for layer in cg.layers():
        CircuitRunner(shift_by_var=False) | layer

    energy_val_list = [val.value() for val in energy_list]

    print(max(energy_val_list), min(energy_val_list))

    verify_ite_for_single_qubit(energy_val_list, hamil, step_size)
