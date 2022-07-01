from copy import copy

from mizore.utils.typed_log import TypedLog
from .valvar import ValVar

from .comp_param import CompParam
from typing import Union, Dict


class CompNode:
    """
    We use CompNode to represent calculations that are
    1. Computational intensity
    2. Not differentiable
    Otherwise we can just use CompParam with certain get_val

    The transpilers should link the CompParam in self.outputs to
    1. The computed data (e.g. quantum backend_circuit simulation result)
    2. The CompParam in self.inputs
    """
    object_counter = 0

    def __init__(self, name=None):
        # The parameters that should be evaluated before the process of the node
        self.inputs: Dict[str, Union[CompParam, ValVar]] = {}
        # The parameters whose value can only be determined by the process of the node
        self.outputs: Dict[str, Union[CompParam, ValVar]] = {}
        self.name = name
        if name is None:
            self.__class__.object_counter += 1
            self.name = self.__class__.__name__ + str(self.__class__.object_counter)
        self.log = TypedLog()

        self.in_graph = True
        self.tags = set()

    def copy_with_map_dict(self, new_elem_dict):
        # TODO test this
        if self in new_elem_dict.keys():
            return new_elem_dict[self]
        new_node = copy(self)
        new_elem_dict[self] = new_node
        # Here we assumed that the node does not have pointer to CompParam other than in
        # outputs and inputs

        # Question: Is that good I don't copy outputs when copying the node
        # The params in outputs can still be copied if they are the inputs of something
        # new_node.outputs = {key: param.copy_with_map_dict(new_elem_dict) for key, param in self.outputs.items()}
        new_node.inputs = {key: param.copy_with_map_dict(new_elem_dict) for key, param in self.inputs.items()}
        new_node.log = TypedLog()  # Clean up the log
        new_node.tags = self.tags.copy()
        return new_node.tags

    def add_input_param(self, key, param):
        self.inputs[key] = param

    def add_output_param(self, key, param: Union[ValVar, CompParam]):
        param.set_home_node(self)
        self.outputs[key] = param
        return param

    def __call__(self, *args, **kwargs):
        return self.outputs

    def __str__(self):
        return self.name
