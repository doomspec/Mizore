from time import time
from typing import List
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.value import Value
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode
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
        for node in node_list:
            args_list.append((node.circuit, node.obs_list, node.params.value(), node.config))
        params_mean = [arg[2] for arg in args_list]
        """
        Important thing when use Pool.
        Jax device array will be casted into numpy array when being passed through processes
        Using numpy array will cause bugs when it operates with Value
        Therefore, we must cast it back to jax array by hand
        """
        with Pool(self.n_proc) as pool:
            exp_vals_and_times = pool.starmap(eval_node_and_time, args_list)
            # Here we cast the array back to Jax array from numpy array
            exp_vals = [to_jax_array(item[0]) for item in exp_vals_and_times]
            for i in range(n_node):
                output_dict[node_list[i]] = {"classical_time": exp_vals_and_times[i][1]}

            if not self.shift_by_var:
                for i in range(len(node_list)):
                    set_expv(node_list[i], exp_vals[i])

            if self.shift_by_var:
                meta_node_list = []
                meta_params_mean = []
                meta_exp_vals = []
                for i in range(len(node_list)):
                    node_i = node_list[i]
                    if isinstance(node_i, DeviceCircuitNode) and node_i.expv_shift_from_var:
                        meta_node_list.append(node_i)
                        meta_params_mean.append(params_mean[i])
                        meta_exp_vals.append(exp_vals[i])
                    else:
                        # If it is a normal QCircuitNode, set its expectation value
                        set_expv(node_list[i], exp_vals[i])
                shift_by_vars = self.eval_shift_by_var(meta_node_list, meta_exp_vals, meta_params_mean, pool)

                for i in range(len(meta_node_list)):
                    # meta_node_list[i].expv.bind_to(exp_vals[i] + shift_by_vals[i])
                    set_expv(meta_node_list[i], exp_vals[i] + shift_by_vars[i].value())
                    # meta_node_list[i].expv.set_value(exp_vals[i] + shift_by_vars[i].value())

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


def set_expv(node: QCircuitNode, value):
    if not node.is_single_obs:
        node.expv.set_value(value)
    else:
        node.expv.set_value(value[0])


def eval_shifted_exps(circuit, obs_list, param, shift, config):
    backend_ops = [BackendOperator(ob) for ob in obs_list]
    config = config if circuit.has_random else None
    shifted_exp_vals = eval_param_shifted_exp_val(circuit, shift, param, backend_ops, config)
    return shifted_exp_vals


def eval_node_and_time(circuit, obs_list, param_mean, config):
    backend_ops = [BackendOperator(ob) for ob in obs_list]
    config = config if circuit.has_random else None
    time_start = time()
    exp_vals = eval_on_param_mean(circuit, param_mean, backend_ops, config)
    time_end = time()
    return exp_vals, (time_end - time_start) * 1e6
