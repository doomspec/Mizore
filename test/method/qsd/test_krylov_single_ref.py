from chemistry.simple_mols import simple_4_qubit_lih
from mizore.backend_circuit.one_qubit_gates import X
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.qsd.krylov_single_ref import H_mat, S_mat, quantum_krylov_single_ref
from mizore.method.qsd.krylov_single_ref_classical import quantum_krylov_single_ref_classical, H_mat_classical, \
    S_mat_classical
from mizore.transpiler.circuit_runner.circuit_runner import CircuitRunner
from numpy.testing import assert_array_almost_equal, assert_allclose
import jax
jax.config.update("jax_enable_x64", True)

def test_H_S_mat():
    ref_circuit = MetaCircuit(n_qubit=4, blocks=[Gates(X(0))])
    hamil = simple_4_qubit_lih()
    del hamil.terms[tuple()]
    delta = 0.2
    for n_basis in range(2, 4):
        H_classical = H_mat_classical(ref_circuit, hamil, n_basis, delta)
        H_quantum = H_mat(ref_circuit, hamil, n_basis, delta)
        S_classical = S_mat_classical(ref_circuit, hamil, n_basis, delta)
        S_quantum = S_mat(ref_circuit, hamil, n_basis, delta)
        cg = H_quantum.build_graph([S_quantum])
        CircuitRunner() | cg
        assert_array_almost_equal(H_quantum.value(), H_classical)
        assert_array_almost_equal(S_quantum.value(), S_classical)


def test_eigvals():
    ref_circuit = MetaCircuit(n_qubit=4, blocks=[Gates(X(0))])
    hamil = simple_4_qubit_lih()
    del hamil.terms[tuple()]
    delta = 0.5
    for n_basis in range(4, 5):
        eigv_classical = quantum_krylov_single_ref_classical(ref_circuit, hamil, n_basis, delta)
        eigvals = quantum_krylov_single_ref(ref_circuit, hamil, n_basis, delta, eps=1e-11)
        CircuitRunner() | eigvals.build_graph()
        assert_allclose(eigv_classical, eigvals.value(), rtol=0.001, atol=0.001)
