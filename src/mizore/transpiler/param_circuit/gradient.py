from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.value import Value

from mizore.transpiler.transpiler import Transpiler
from jax.numpy import transpose
from mizore import jax_array


class GradientCircuit(Transpiler):
    def __init__(self, add_grad_to_graph=False, init_shot_num=10000):
        self.add_grad_to_graph = add_grad_to_graph
        self.init_shot_num = init_shot_num
        super().__init__()

    def transpile(self, graph_iterator: GraphIterator):
        output_dict = {}
        for pqcnode in graph_iterator.by_type(DeviceCircuitNode):
            pqcnode: DeviceCircuitNode
            origin_bc = pqcnode.circuit
            n_param = origin_bc.n_param
            param_grads = []
            n_obs = len(pqcnode.obs_list)
            for param_i in range(n_param):
                shifted_bc = origin_bc.get_gradient_circuits(param_i)
                param_grad = Value(jax_array([0.0] * n_obs)) if not pqcnode.is_single_obs else Value(0.0)
                shift_i = 0
                for coeff, bc in shifted_bc:
                    new_pqc_node = DeviceCircuitNode(bc, pqcnode.obs,
                                                     name=f"{pqcnode.name}-Param-{param_i}-Grad-{shift_i}")
                    new_pqc_node.params.bind_to(pqcnode.params)
                    new_pqc_node.shot_num.set_value(self.init_shot_num)
                    param_grad = param_grad + coeff * new_pqc_node()
                    shift_i += 1
                param_grads.append(param_grad)
            param_grads = Value.unary_operator(Value.array(param_grads), transpose)
            param_grads.const_approx = True
            param_grads.name = f"{pqcnode.name}-ParamGrads"
            output_dict[pqcnode] = param_grads
        if self.add_grad_to_graph:
            graph_iterator.comp_graph.add_output_vals(list(output_dict.values()), reconstruct=True)
        return output_dict
