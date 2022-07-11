from typing import Union, Tuple

from mizore.comp_graph.value import Value
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.inner_product_circuits import C_mat_imag, A_mat_real
from mizore.operators import QubitOperator
from jax.numpy.linalg import lstsq


def real_evol_gradient(circuit: MetaCircuit, hamil: QubitOperator, param: Union[Value, None] = None) -> Tuple[Value, Value, Value]:
    """
    Get the gradient of the params in the circuit for the closest move to the evolution e^{iHt}
    Notice that, the constant term in H will be omitted!
    :return: x, A_real, C_imag
    """
    if param is None:
        param = Value([0.0] * circuit.n_param)

    A = A_mat_real(circuit, param)
    C = C_mat_imag(circuit, hamil, param)

    # Here we use -lstsq instead of lstsq because we are simulating e^{iHt}
    return Value.binary_operator(A, C, lambda A_, b: -lstsq(A_, b)[0]), A, C


def get_residue_imag_evol(A_real: Value, C_imag: Value):
    A_value = A_real.value()
    C_value = C_imag.value()

    def residue_fun(A_, b_):
        return A_value @ lstsq(A_, b_)[0] - C_value

    residue = Value.binary_operator(A_real, C_imag, residue_fun)
    return residue