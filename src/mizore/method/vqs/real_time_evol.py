from typing import Union, Tuple, Dict, Optional

from mizore.comp_graph.node.qc_node import QCircuitNode
from mizore.comp_graph.value import Value
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.method.vqs.inner_product_circuits import C_mat_imag_rte, A_mat_real
from mizore.operators import QubitOperator
from jax.numpy.linalg import lstsq
import jax.numpy as jnp


def real_evol_gradient(circuit: MetaCircuit, hamil: QubitOperator, param: Optional[Value] = None,
                       rcond=None, calc_Delta=False) -> Tuple[Value, Dict]:
    """
    Get the gradient of the params in the circuit for the closest move to the evolution e^{iHt}
    Notice that, the constant term in H will be omitted!
    :return: x, res_dict
    """
    if param is None:
        param = Value([0.0] * circuit.n_param)

    hamil_no_const, const = hamil.remove_constant()

    A = A_mat_real(circuit, param)
    C = C_mat_imag_rte(circuit, hamil_no_const, param)

    # Here we use lstsq because we are simulating e^{-iHt}
    evol_grad = Value.binary_operator(A, C, lambda A_, b: lstsq(A_, b, rcond=rcond)[0])

    res_dict = {"A": A, "C": C}

    if calc_Delta:
        hamil_sqr = hamil_no_const * hamil_no_const
        hamil_sqr.compress()
        hamil_sqr_node = QCircuitNode(circuit, hamil_sqr, name="HamilSqr")
        Delta_sqr = evol_grad.dot(Value.binary_operator(A.value(), evol_grad, jnp.dot)) \
                    - 2 * evol_grad.dot(C.value()) + hamil_sqr_node()
        Delta = jnp.sqrt | Delta_sqr
        res_dict["Delta"] = Delta

    return evol_grad, res_dict


def get_residue_imag_evol(A_real: Value, C_imag: Value):
    A_value = A_real.value()
    C_value = C_imag.value()

    def residue_fun(A_, b_):
        return A_value @ lstsq(A_, b_)[0] - C_value

    residue = Value.binary_operator(A_real, C_imag, residue_fun)
    return residue
