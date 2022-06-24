from typing import Tuple, Iterable

from mizore.comp_graph.comp_graph import CompGraph, GraphIterator
from mizore.comp_graph.comp_node import CompNode


class Transpiler:
    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__
        self.types = []

    def transpiler_param_output(self):
        return None

    def __rshift__(self, other):
        return self.transpile([other])

    def __or__(self, other):
        if isinstance(other, GraphIterator):
            return self.transpile(other)
        elif isinstance(other, CompGraph):
            return self.transpile(other.all())
        elif isinstance(other, Tuple):
            if isinstance(other[0], CompNode):
                return self.transpile(GraphIterator([other[0]], other[1]))
            elif isinstance(other[0], Iterable):
                return self.transpile(GraphIterator(other[0], other[1]))
            else:
                raise TypeError()
        elif isinstance(other, CompNode):
            return self.transpile(GraphIterator([other], None))
        else:
            raise TypeError()

    def transpile(self, other):
        pass

    def __str__(self):
        return type(self).__name__
