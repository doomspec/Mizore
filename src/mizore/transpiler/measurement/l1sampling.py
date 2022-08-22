from typing import Union, Dict

from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.comp_graph.parameter import Parameter
from mizore.comp_graph.value import Value
from mizore.transpiler.transpiler import Transpiler
import jax.numpy as jnp


class L1Sampling(Transpiler):

    def __init__(self, state_ignorant=False, name=None, default_shot_num=10000,
                 shot_allocate: Union[Dict, None] = None):
        Transpiler.__init__(self, name)
        self.state_ignorant = state_ignorant
        self.default_shot_num = default_shot_num
        self.shot_allocate: Dict = shot_allocate if shot_allocate is not None else {}

    def transpile(self, graph_iterator: GraphIterator):
        output_dict = {}
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            expv_list = node.expv_list()
            shot_num_list = []

            if node in self.shot_allocate:
                node_shot_allocate = self.shot_allocate[node]
                for i_expv in range(len(expv_list)):
                    shot_num_i = node.add_output_value(f"ShotNum-{i_expv}", Parameter(node_shot_allocate[i_expv]))
                    shot_num_list.append(shot_num_i)
            else:
                for i_expv in range(len(expv_list)):
                    shot_num_i = node.add_output_value(f"ShotNum-{i_expv}", Parameter(self.default_shot_num))
                    shot_num_list.append(shot_num_i)

            for i_expv in range(len(expv_list)):
                var_coeff = (node.obs_list[i_expv].get_l1_norm_omit_const()) ** 2
                if not self.state_ignorant:
                    var_coeff -= (expv_list[i_expv].value() - node.obs_list[i_expv].constant) ** 2
                expv_list[i_expv].set_to_random_variable(var_coeff / shot_num_list[i_expv], check_valid=False)

            node.shot_num_total.bind_to(Value(args=shot_num_list, operator=jnp.sum))
            node.shot_num_overwritten = shot_num_l1_sampling

        return output_dict


def shot_num_l1_sampling(node: DeviceCircuitNode):
    if not node.is_single_obs:
        shot_nums = []
        for i_expv in range(len(node.obs_list)):
            shot_nums.append(node.outputs[f"ShotNum-{i_expv}"])
        return shot_nums
    else:
        return node.outputs["ShotNum-0"]

