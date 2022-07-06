from mizore.comp_graph.value import Value
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.inner_product_circuits import C_mat_imag, A_mat_real
from mizore.operators import QubitOperator
from jax.numpy.linalg import lstsq


def real_evol_gradient(circuit: MetaCircuit, hamil: QubitOperator) -> Value:
    A = A_mat_real(circuit)
    C = C_mat_imag(circuit, hamil)
    return Value.binary_operator(A, C, lambda A, b: lstsq(A, b)[0])
