from chemistry.simple_mols import simple_4_qubit_lih
from mizore.backend_circuit.one_qubit_gates import X
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.block.trotter import Trotter
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
from numpy.testing import assert_allclose


def get_innerp_list(init_circuit: MetaCircuit, hamil: QubitOperator, n_steps, delta_t):
    n_qubit = init_circuit.n_qubit
    exact_evol = ExactEvolution(hamil)
    ref_circuit = MetaCircuit(n_qubit, blocks=init_circuit.block_list + [exact_evol])

    innerp_list = []
    for n_step in n_steps:
        trotter_block = Trotter(hamil, delta_t, delta_t*n_step)
        circuit = MetaCircuit(n_qubit, blocks=init_circuit.block_list + [trotter_block])
        state = circuit.get_backend_state()
        state_expected = ref_circuit.get_backend_state([n_step * delta_t])
        innerp = state.inner_product(state_expected)
        innerp_list.append(innerp)

    return innerp_list


def test_single_rotation():
    qset = [0, 1, 2]
    pauli = [1, 3, 2]
    hamil = QubitOperator.from_qset_op(qset, pauli) + 1
    n_qubit = 3
    max_n_step = 10
    init_circuit = MetaCircuit(n_qubit, blocks=[Gates(X(0))])
    innerp_list = get_innerp_list(init_circuit, hamil, range(1, max_n_step), 0.5)
    assert_allclose(innerp_list, [1.0] * (max_n_step - 1))


def test_lih_hamil():
    hamil = simple_4_qubit_lih()
    init_circuit = MetaCircuit(4, blocks=[Gates(X(0))])
    max_n_step = 10
    innerp_list = get_innerp_list(init_circuit, hamil, range(1, max_n_step), 0.1)
    assert_allclose(innerp_list, [1.0] * (max_n_step - 1), rtol=1e-5)
