from typing import List, Dict
from qulacs import QuantumState, DensityMatrix
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore import np_array, jax_array, to_jax_array, to_np_array
from mizore.comp_graph.comp_param import CompParam
from mizore.comp_graph.valvar import ValVar
# import numpy as np
import jax.numpy as jnp

Variance_To_Neglect = 1e-12


def eval_on_param_mean(circuit: MetaCircuit, params, qulacs_ops, random_config=None):
    qulacs_circuit = circuit.get_backend_circuit(params)
    # Run and update the state
    res: List[complex]
    if random_config is None:
        state = QuantumState(circuit.n_qubit)
        qulacs_circuit.update_quantum_state(state)
        res = get_exp_value_list(qulacs_ops, state)
    else:  # When the task is probabilistic
        random_config: Dict
        if random_config.get("use_dm", False):
            state = DensityMatrix(circuit.n_qubit)
            qulacs_circuit.update_quantum_state(state)
            res = get_exp_value_list(qulacs_ops, state)
        else:
            n_exp = random_config["n_exp"]
            if n_exp == 1:
                print("Warning: number of experiments is only 1")
            res_list = [None] * n_exp
            for i in range(n_exp):
                # TODO This process can be improved a lot
                # See https://arxiv.org/abs/1904.11590
                state = QuantumState(circuit.n_qubit)
                qulacs_circuit.update_quantum_state(state)
                res_list[i] = get_exp_value_list(qulacs_ops, state)
                del state
            res_list = jax_array(res_list)
            res = jnp.average(res_list, axis=0)
    for i in range(len(res)):
        if abs(res[i].imag) > 1e-6:
            raise NotImplementedError("Complex expectation value is not yet supported")
        res[i] = res[i].real

    return jax_array(res)


def get_exp_value_list(ops, state):
    return [op.get_expectation_value(state) for op in ops]

def eval_param_shifted_exp_val(circuit: MetaCircuit, shift, params, qulacs_ops, random_config=None):
    shifted_exps = []
    for param_i in range(circuit.n_param):
        # If the variance is too small, we skip the calculation
        # if params_var[param_i] < Variance_To_Neglect:
        #    second_grad_list.append([0.0] * len(qulacs_ops))
        #    continue
        params[param_i] += shift
        circuit.temp_params = to_np_array(params)
        exp_val = eval_on_param_mean(circuit, params, qulacs_ops,
                                             random_config=random_config)
        params[param_i] -= shift
        shifted_exps.append(exp_val)
    return jax_array(shifted_exps)

def eval_second_grads(circuit: MetaCircuit, exp_on_param_mean, params, qulacs_ops, eps=1e-4, random_config=None):
    second_grad_list = []
    for param_i in range(circuit.n_param):
        # If the variance is too small, we skip the calculation
        # if params_var[param_i] < Variance_To_Neglect:
        #    second_grad_list.append([0.0] * len(qulacs_ops))
        #    continue
        params[param_i] += eps
        circuit.temp_params = to_np_array(params)
        exp_val_forward = eval_on_param_mean(circuit, params, qulacs_ops,
                                             random_config=random_config)
        params[param_i] -= 2 * eps
        circuit.temp_params = to_np_array(params)
        exp_val_backward = eval_on_param_mean(circuit, params, qulacs_ops,
                                              random_config=random_config)
        params[param_i] += eps
        # grads = (exp_val_forward - exp_val_backward) / (2 * eps)
        second_grads = (exp_val_forward - 2 * exp_on_param_mean + exp_val_backward) / (eps ** 2)
        # grad_list.append(grads)
        second_grad_list.append(second_grads)

    # grad_list = np_array(grad_list).transpose()
    second_grad_list = to_jax_array(second_grad_list).transpose()
    # Add the variance's contribution to the mean shift
    # exp_val_contributed_by_var = CompParam.unary_operator((second_grad_list * params_.var / 2), lambda x: jnp.sum(x, axis=1))

    # exp_val_var = (grad_list ** 2) * params_var - (second_grad_list ** 2) * (params_var ** 2) / 4
    # exp_val_var = np.sum(exp_val_var, axis=1)

    # exp_val_contributed_by_var = jnp.sum((second_grad_list * params_.var.value() / 2), axis=1)

    return second_grad_list
