
from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.mc_node import MetaCircuitNode
from mizore.comp_graph.valvar import ValVar

from mizore.transpiler.transpiler import Transpiler
from jax.numpy import transpose
from mizore import jax_array
class GradientCircuit(Transpiler):
    def __init__(self, add_grad_to_graph = False, init_shot_num=10000):
        self.add_grad_to_graph = add_grad_to_graph
        self.init_shot_num = init_shot_num
        super().__init__()

    def transpile(self, graph_iterator: GraphIterator):
        output_dict = {}
        for pqcnode in graph_iterator.by_type(MetaCircuitNode):
            pqcnode: MetaCircuitNode
            origin_bc = pqcnode.circuit
            n_param = origin_bc.n_param
            grad_valvars = []
            is_single_obs = len(pqcnode.obs) == 1
            n_obs = len(pqcnode.obs)
            for param_i in range(n_param):
                shifted_bc = origin_bc.get_gradient_circuits(param_i)
                grad_valvar = ValVar(jax_array([0.0]*n_obs), jax_array([0.0]*n_obs))
                shift_i = 0
                for coeff, bc in shifted_bc:
                    new_pqc_node = MetaCircuitNode(bc, pqcnode.obs, name=f"{pqcnode.name}-Param-{param_i}-Grad-{shift_i}")
                    new_pqc_node.params.set_value(pqcnode.params)
                    new_pqc_node.shot_num.set_value(self.init_shot_num)
                    grad_valvar = grad_valvar + coeff * new_pqc_node()
                    shift_i += 1
                grad_valvars.append(grad_valvar)
            output = ValVar.array(grad_valvars).simple_unary_op(transpose)
            if is_single_obs:
                output_dict[pqcnode] = output.get_by_index(0)
            else:
                output_dict[pqcnode] = output
        if self.add_grad_to_graph:
            graph_iterator.comp_graph.add_output_elems(list(output_dict.values()), reconstruct=True)
        return output_dict