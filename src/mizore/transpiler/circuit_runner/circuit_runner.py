from time import time
from typing import List, Dict
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.value import Value
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode, set_node_expv_list
from mizore.backend_circuit.backend_circuit import BackendOperator
from mizore.transpiler.circuit_runner._circuit_runner_impl import eval_on_param_mean, \
    eval_param_shifted_exp_val
from mizore.transpiler.transpiler import Transpiler

from mizore import to_jax_array
from multiprocessing import Pool
import jax.numpy as jnp


class CircuitRunner(Transpiler):
    def __init__(self, n_proc=4, shift_by_var=False):
        super().__init__()
        self.n_proc = n_proc
        self.eps = 1e-4
        self.shift_by_var = shift_by_var  # this option is buggy

    def transpile(self, target_nodes: GraphIterator):
        output_dict = {}
        node_list: List[QCircuitNode] = list(target_nodes.by_type(QCircuitNode))
        n_node = len(node_list)
        args_list = []
        params_means = []
        aux_obs_indices_list = []
        for node in node_list:
            aux_obs_list, aux_obs_indices = flatten_obs_dict(node.aux_obs_dict)
            aux_obs_indices_list.append(aux_obs_indices)
            node_params = node.params.value()
            params_means.append(node_params)
            args_list.append((node.circuit, node.obs_list+aux_obs_list, node_params, node.config))

        """
        Important thing when use Pool.
        Jax device array will be casted into numpy array when being passed through processes
        Using numpy array will cause bugs when it operates with Value
        Therefore, we must cast it back to jax array by hand
        """
        with Pool(self.n_proc) as pool:
            exp_vals_times_process_res = pool.starmap(eval_node, args_list)
            # Here we cast the array back to Jax array from numpy array
            exp_vals_res = [to_jax_array(item[0]) for item in exp_vals_times_process_res]
            for i in range(n_node):
                output_dict[node_list[i]] = {"classical_time": exp_vals_times_process_res[i][1]}

            if not self.shift_by_var:
                for i in range(len(node_list)):
                    aux_obs_indices = aux_obs_indices_list[i]
                    set_expv(node_list[i], exp_vals_res[i], aux_obs_indices)

            if self.shift_by_var:
                """
                This is actually doable
                We just need to ensure the variables are mutually independent
                """
                """
                meta_node_list = []
                meta_params_mean = []
                meta_exp_vals = []
                for i in range(len(node_list)):
                    node_i = node_list[i]
                    if isinstance(node_i, DeviceCircuitNode) and node_i.expv_shift_from_var:
                        meta_node_list.append(node_i)
                        meta_params_mean.append(params_means[i])
                        meta_exp_vals.append(exp_vals_res[i])
                    else:
                        # If it is a normal QCircuitNode, set its expectation value
                        set_expv(node_list[i], exp_vals_res[i])
                shift_by_vars = self.eval_shift_by_var(meta_node_list, meta_exp_vals, meta_params_mean, pool)

                for i in range(len(meta_node_list)):
                    # meta_node_list[i].expv.bind_to(exp_vals[i] + shift_by_vals[i])
                    set_expv(meta_node_list[i], exp_vals_res[i] + shift_by_vars[i].value())
                    # meta_node_list[i].expv.set_value(exp_vals[i] + shift_by_vars[i].value())
                """
        return output_dict

    def eval_shift_by_var(self, node_list, exp_vals, params_mean, pool):
        n_node = len(node_list)
        args_forward = [(node_list[i].circuit, node_list[i].obs_list, params_mean[i], self.eps,
                         node_list[i].config) for i in range(n_node)]
        args_backward = [(node_list[i].circuit, node_list[i].obs_list, params_mean[i], -self.eps,
                          node_list[i].config) for i in range(n_node)]
        args_second_grad = args_forward + args_backward
        # with Pool(self.n_proc) as pool:
        shifted_exp_vals = pool.starmap(eval_shifted_exps, args_second_grad)
        shift_by_vars = []
        for i in range(n_node):
            second_grad = (shifted_exp_vals[i] - 2 * exp_vals[i] + shifted_exp_vals[i + n_node]) / (self.eps ** 2)
            second_grad = jnp.transpose(second_grad)
            node: DeviceCircuitNode = node_list[i]
            if node.circuit.n_param != 0:
                # shift_by_var = Value.unary_operator((node.params.var * second_grad / 2), lambda x: jnp.sum(x, axis=1))
                shift_by_var = Value(jnp.sum(node.params.var.value() * second_grad / 2, axis=1))  # Non-differentiable
            else:
                shift_by_var = Value(0.0)

            shift_by_vars.append(shift_by_var)
        return shift_by_vars


def flatten_obs_dict(obs_dict: Dict):
    obs_list = []
    key_index = {}
    for key, value in obs_dict.items():
        obs = value["obs"]
        key_index[key] = len(obs_list)
        obs_list.extend(obs)
    return obs_list, key_index


def assign_res_to_obs_dict(res_list, key_index: Dict, obs_dict):
    for key, starting_index in key_index.items():
        obs_item = obs_dict[key]
        obs_item["res"] = res_list[starting_index:starting_index + len(obs_item["obs"])]


def set_expv(node: QCircuitNode, expv_res, aux_obs_indices):
    main_len = len(node.obs_list)
    expv_main = expv_res[:main_len]
    set_node_expv_list(node, expv_main)
    if main_len < len(expv_res):
        assign_res_to_obs_dict(expv_res[main_len:], aux_obs_indices, node.aux_obs_dict)


def eval_shifted_exps(circuit, obs_list, param, shift, config):
    backend_ops = [BackendOperator(ob) for ob in obs_list]
    config = config if circuit.has_random else None
    shifted_exp_vals = eval_param_shifted_exp_val(circuit, shift, param, backend_ops, config)
    return shifted_exp_vals


def eval_node(circuit, obs_list, param_mean, config):
    backend_ops = [BackendOperator(ob) for ob in obs_list]
    config = config if circuit.has_random else None
    time_start = time()
    exp_vals = eval_on_param_mean(circuit, param_mean, backend_ops, config)
    time_end = time()
    return exp_vals, (time_end - time_start) * 1e6
