from time import time
from typing import List
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.comp_param import CompParam
from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.comp_graph.node.qc_node import QCircuitNode

from mizore.transpiler.circuit_runner._circuit_runner_impl import eval_on_param_mean, eval_second_grads, \
    eval_param_shifted_exp_val
from mizore.transpiler.transpiler import Transpiler

from mizore import to_jax_array
from multiprocessing import Pool
import jax.numpy as jnp


class CircuitRunner(Transpiler):
    def __init__(self, cache_key=None, n_proc=4):
        super().__init__()
        self.n_proc = n_proc
        self.eps = 1e-4
        self.shift_by_var = True
        self.cache_key = cache_key

    def transpile(self, target_nodes: GraphIterator):
        output_dict = {}
        node_list: List[MetaCircuitNode] = list(target_nodes.by_type(MetaCircuitNode))
        n_node = len(node_list)
        args_list = [(node.circuit, node.obs, node.params.mean.get_value(cache_key=self.cache_key),
                      node.random_config) for node in node_list]

        params_mean = [arg[2] for arg in args_list]
        """
        Important thing when use Pool.
        Jax device array will be casted into numpy array when being passed through processes
        Using numpy array will cause bugs when it operates with CompParam
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
                    node_list[i].exp_mean.set_value(exp_vals[i])

            if self.shift_by_var:
                shift_by_vals = self.eval_shift_by_var(node_list, exp_vals, params_mean, pool)
                for i in range(len(node_list)):
                    node_list[i].exp_mean.set_value(exp_vals[i] + shift_by_vals[i])
                    node_list[i].exp_mean.eval_and_cache(self.cache_key)


        return output_dict

    def eval_shift_by_var(self, node_list, exp_vals, params_mean, pool):
        n_node = len(node_list)
        args_forward = [(node_list[i].circuit, node_list[i].obs, params_mean[i], self.eps,
                         node_list[i].random_config) for i in range(n_node)]
        args_backward = [(node_list[i].circuit, node_list[i].obs, params_mean[i], -self.eps,
                          node_list[i].random_config) for i in range(n_node)]
        args_second_grad = args_forward + args_backward
        # with Pool(self.n_proc) as pool:
        shifted_exp_vals = pool.starmap(eval_shifted_exps, args_second_grad)
        shift_by_vars = []
        for i in range(n_node):
            second_grad = (shifted_exp_vals[i] - 2 * exp_vals[i] + shifted_exp_vals[i + n_node]) / (self.eps ** 2)
            second_grad = to_jax_array(second_grad.transpose())
            node: MetaCircuitNode = node_list[i]
            if node.circuit.n_param != 0:
                shift_by_var = CompParam.unary_operator((node.params.var * second_grad / 2), lambda x: jnp.sum(x, axis=1))
                # shift_by_var = jnp.sum(node.params.var.value() * second_grad / 2, axis=1) # Non-differentiable
            else:
                shift_by_var = 0

            shift_by_vars.append(shift_by_var)
        return shift_by_vars


def eval_shifted_exps(circuit, obs, param, shift, random_config):
    backend_ops = [ob.get_backend_operator() for ob in obs]
    random_config = random_config if circuit.has_random else None
    shifted_exp_vals = eval_param_shifted_exp_val(circuit, shift, param, backend_ops, random_config)
    return shifted_exp_vals


def eval_node_and_time(circuit, obs, param_mean, random_config):
    backend_ops = [ob.get_backend_operator() for ob in obs]
    random_config = random_config if circuit.has_random else None
    time_start = time()
    exp_vals = eval_on_param_mean(circuit, param_mean, backend_ops, random_config)
    time_end = time()
    return (exp_vals, (time_end - time_start) * 1e6)


"""
for i in range(n_node):
    backend_ops = [ob.get_backend_operator() for ob in node_list[i].obs]
    second_grads_test = eval_second_grads(node_list[i].backend_circuit, exp_vals[i], param_mean[i].tolist(), backend_ops, random_config=node_list[i].random_config)
print(second_grads_test-second_grads)
"""
