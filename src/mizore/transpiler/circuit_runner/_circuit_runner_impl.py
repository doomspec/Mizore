from typing import List, Dict
from mizore.backend_circuit.backend_circuit import BackendState
from mizore.backend_circuit.backend_op import BackendOperator
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore import jax_array, to_jax_array, to_np_array
# import numpy as np
import jax.numpy as jnp

from mizore.transpiler.circuit_runner.state_processor import StateProcessor


def eval_on_param_mean(circuit: MetaCircuit, params, backend_ops: List[BackendOperator], config=None,
                       state_processors: List[StateProcessor] = None):
    backend_circuit = circuit.get_backend_circuit(params)
    # Run and update the state
    res: List[complex]
    state_processor_res = {}
    if config is None:
        backend_state = backend_circuit.get_quantum_state(dm=False)
        res = backend_state.get_many_expv(backend_ops)
        use_post_processors(backend_state, state_processors, state_processor_res)
    else:  # When the task is probabilistic
        config: Dict
        if config.get("use_dm", False):  # If use_dm = True
            backend_state = backend_circuit.get_quantum_state(dm=True)
            res = backend_state.get_many_expv(backend_ops)
            use_post_processors(backend_state, state_processors, state_processor_res)
        else:
            assert state_processors is None
            n_exp = config["n_exp"]
            if n_exp == 1:
                print("Warning: number of experiments is only 1")
            res_list = [None] * n_exp
            for i in range(n_exp):
                # TODO This process can be improved a lot
                # See https://arxiv.org/abs/1904.11590
                backend_state = backend_circuit.get_quantum_state(dm=False)
                res_list[i] = backend_state.get_many_expv(backend_ops)
            res_list = jax_array(res_list)
            res = jnp.average(res_list, axis=0)
    for i in range(len(res)):
        if abs(res[i].imag) > 1e-6:
            raise NotImplementedError("Complex expectation value is not yet supported")
        res[i] = res[i].real
    return jax_array(res), state_processor_res


def use_post_processors(backend_state, state_processors, state_processor_res):
    processor: StateProcessor
    for processor in state_processors:
        res = processor.process(backend_state)
        state_processor_res[processor.key] = res


def get_exp_value_list(ops, state):
    return [op.get_expectation_value(state) for op in ops]


def eval_param_shifted_exp_val(circuit: MetaCircuit, shift, params, backend_ops: List[BackendOperator], config=None):
    # TODO this is buggy!!!!!
    shifted_exps = []
    for param_i in range(circuit.n_param):
        params[param_i] += shift
        circuit.temp_params = to_np_array(params)
        exp_val = eval_on_param_mean(circuit, params, backend_ops,
                                     config=config)
        params[param_i] -= shift
        shifted_exps.append(exp_val)
    return to_jax_array(shifted_exps)


def eval_second_grads(circuit: MetaCircuit, exp_on_param_mean, params,
                      backend_ops: List[BackendOperator], eps=1e-4, config=None):
    second_grad_list = []
    for param_i in range(circuit.n_param):
        # If the variance is too small, we skip the calculation
        # if params_var[param_i] < Variance_To_Neglect:
        #    second_grad_list.append([0.0] * len(backend_ops))
        #    continue
        params[param_i] += eps
        circuit.temp_params = to_np_array(params)
        exp_val_forward = eval_on_param_mean(circuit, params, backend_ops,
                                             config=config)
        params[param_i] -= 2 * eps
        circuit.temp_params = to_np_array(params)
        exp_val_backward = eval_on_param_mean(circuit, params, backend_ops,
                                              config=config)
        params[param_i] += eps
        # grads = (exp_val_forward - exp_val_backward) / (2 * eps)
        second_grads = (exp_val_forward - 2 * exp_on_param_mean + exp_val_backward) / (eps ** 2)
        # grad_list.append(grads)
        second_grad_list.append(second_grads)

    second_grad_list = to_jax_array(second_grad_list).transpose()
    # second_grad_list = [list(item) for item in zip(*second_grad_list)] # Transpose the list

    return second_grad_list
