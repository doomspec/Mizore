from typing import Union, Tuple

from mizore.comp_graph.value import Value
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.inner_product_circuits import C_mat_imag_rte, A_mat_real, C_mat_real_ite
from mizore.operators import QubitOperator
from jax.numpy.linalg import lstsq, solve


def imag_evol_gradient(circuit: MetaCircuit, hamil: QubitOperator, param: Union[Value, None] = None, rcond=None) -> \
        Tuple[
            Value, Value, Value, Value]:
    """
    Get the gradient of the params in the circuit for the closest move to the evolution e^{-Ht}
    :return: x, A_real, C_real
    """
    if param is None:
        param = Value([0.0] * circuit.n_param)

    A = A_mat_real(circuit, param)
    C, current_energy = C_mat_real_ite(circuit, hamil, param)

    evol_grad = Value.binary_operator(A, C, lambda A_, b: lstsq(A_, -b, rcond=rcond)[0])
    # evol_grad = Value.binary_operator(A, C, lambda A_, b: solve(A_, -b))
    evol_grad.name = "ITE-Grad"
    # Here we use -b instead of lstsq because we are solving Ax=-C
    return evol_grad, A, C, current_energy


def get_residue_imag_evol(A_real: Value, C_real: Value, rcond=None):
    A_value = A_real.value()
    C_value = C_real.value()

    def residue_fun(A_, b_):
        # we are solving Ax=-C, so the residue is Ax+C
        return A_value @ lstsq(A_, b_, rcond=rcond)[0] + C_value

    residue = Value.binary_operator(A_real, C_real, residue_fun)
    return residue
