import numbers

from mizore.utils.type_check import is_number
from .value import Value


class Parameter(Value):
    def __init__(self, init_val, name=None):
        super().__init__(name=name)
        self.set_value(init_val)
        # When set_value is call and the param is linked to another param, is_parameter will be turned to False.
        self.is_parameter = True
