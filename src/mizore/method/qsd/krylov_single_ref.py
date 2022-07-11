from typing import Union, List

from mizore.backend_circuit.multi_qubit_gates import PauliGate
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.block.exact_evolution import ExactEvolution
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.basic.inner_product import get_inner_prod_by_additional
from mizore.method.qsd.generalized_eigv import generalized_eigv_by_wang
from mizore.operators import QubitOperator
from jax.scipy.linalg import eigh as eigh_scipy
from jax.numpy.linalg import eigh as eigh_numpy
import jax.numpy as jnp
import jax


def H_mat_term_diagonal(ref_circuit: MetaCircuit, hamil: QubitOperator, t: float) -> Value:
    assert t >= 0.0
    circuit = ref_circuit.replica()
    if t != 0.0:
        circuit.add_blocks([ExactEvolution(hamil, init_time=t, to_decompose=True)])
    node = DeviceCircuitNode(circuit, hamil, name="QSD-H-diag")
    return node()


def H_mat_term(ref_circuit: MetaCircuit, hamil: QubitOperator, t1: float, t2: float) -> Value:
    assert t2 > t1

    circuit0 = ref_circuit.replica()
    if t1 != 0.0:
        circuit0.add_blocks([ExactEvolution(hamil, init_time=t1, to_decompose=True)])

    summand_list = []
    for qset, op, weight in hamil.qset_op_weight_omit_const():
        additional = [ExactEvolution(hamil, init_time=t2 - t1, to_decompose=True), Gates(PauliGate(qset, op))]
        summand = weight * get_inner_prod_by_additional(circuit0, additional)
        summand_list.append(summand)

    res = Value(args=[Value.array(summand_list)], operator=jnp.sum)

    return res


def S_mat_term(ref_circuit: MetaCircuit, hamil: QubitOperator, t1: float, t2: float) -> Value:
    assert t2 > t1

    circuit0 = ref_circuit.replica()
    if t1 != 0.0:
        circuit0.add_blocks([ExactEvolution(hamil, init_time=t1, to_decompose=True)])

    innerp = get_inner_prod_by_additional(circuit0, [ExactEvolution(hamil, init_time=t2 - t1, to_decompose=True)])

    return innerp


def H_mat(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float) -> Value:
    assert n_basis > 1
    H: List[List[Union[Value, None]]] = [[None for _ in range(n_basis)] for _ in range(n_basis)]
    for i in range(n_basis):
        H[i][i] = H_mat_term_diagonal(ref_circuit, hamil, delta * i)
        for j in range(i + 1, n_basis):
            H[i][j] = H_mat_term(ref_circuit, hamil, delta * i, delta * j)
            H[j][i] = H[i][j].conjugate()
        H[i] = Value.array(H[i])
    return Value.array(H)


def S_mat(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float) -> Value:
    assert n_basis > 1
    S: List[List[Union[Value, None]]] = [[None for _ in range(n_basis)] for _ in range(n_basis)]
    for i in range(n_basis):
        S[i][i] = Value(1.0 + 0.0j)
        for j in range(i + 1, n_basis):
            S[i][j] = S_mat_term(ref_circuit, hamil, delta * i, delta * j)
            S[j][i] = S[i][j].conjugate()
        S[i] = Value.array(S[i])
    return Value.array(S)


def quantum_krylov_single_ref(ref_circuit: MetaCircuit, hamil: QubitOperator, n_basis: int, delta: float, eps=1e-10) -> Value:
    if not jax.config.values["jax_enable_x64"]:
        raise Exception("Generalized eigenvalue solver might be very unstable if x64 precision is not enabled. Consider"
                        "calling \njax.config.update(\"jax_enable_x64\", True)\nto enable it.")

    H = H_mat(ref_circuit, hamil, n_basis, delta)
    S = S_mat(ref_circuit, hamil, n_basis, delta)
    """
    def generalized_eigvals(H_, S_):
        return eigh_scipy(H_, S_, eigvals_only=True)
    """

    def generalized_eigvals(H_, S_):
        return generalized_eigv_by_wang(H_, S_, eigvals_only=True, eps=eps)

    return Value.binary_operator(H, S, generalized_eigvals)
