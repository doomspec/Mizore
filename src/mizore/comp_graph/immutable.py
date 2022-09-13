from mizore.comp_graph.value import Value
from mizore.utils.type_check import is_number


class Immutable(Value):
    def __init__(self, init_val=None, name=None, home_node=None):
        super().__init__(val=init_val, name=name, home_node=home_node)

    def set_value(self, val):
        if is_number(val):
            self.args = []
            self.operator = lambda: val
        else:
            raise Exception("set_value by Value is not valid for Immutable. Use bind_to instead.")