from copy import copy

from mizore.utils.typed_log import TypedLog
from .immutable import Immutable

from .value import Value
from typing import Union, Dict
from jax.numpy import ndarray


class CompNode:
    """
    We use CompNode to represent calculations that are
    1. Computational intensity
    2. Not differentiable
    Otherwise we can just use Value with certain get_val

    The transpilers should link the Value in self.outputs to
    1. The computed data (e.g. quantum backend_circuit simulation result)
    2. The Value in self.inputs
    """
    object_counter = 0

    def __init__(self, name=None):
        # The values that should be evaluated before the process of the node
        self.inputs: Dict[str, Value] = {}
        # The values that can only be determined by the process of the node
        self.outputs: Dict[str, Value] = {}
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
        # Here we assumed that the node does not have pointer to Value other than in
        # outputs and inputs

        # Question: Is that good I don't copy outputs when copying the node
        # The values in outputs can still be copied if they are the inputs of something
        # new_node.outputs = {key: val.copy_with_map_dict(new_elem_dict) for key, val in self.outputs.items()}
        new_node.inputs = {key: val.copy_with_map_dict(new_elem_dict) for key, val in self.inputs.items()}
        new_node.log = TypedLog()  # Clean up the log
        new_node.tags = self.tags.copy()
        return new_node.tags

    def add_input_value(self, key, val: Union[Value, None, ndarray] = None, only_bind=False):
        new_input = Immutable(val)
        new_input.name = f"{self.name}-{key}"
        self.inputs[key] = new_input
        return new_input

    def add_output_value(self, key, val: Union[Value, None, ndarray] = None):
        new_output = Immutable(val)
        new_output.name = f"{self.name}-{key}"
        new_output.set_home_node(self)
        self.outputs[key] = new_output
        return new_output

    def __call__(self, *args, **kwargs):
        return self.outputs

    def __str__(self):
        return self.name
