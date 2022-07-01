from __future__ import annotations

import numbers
from copy import copy
from typing import Union, List, Iterable

from mizore.comp_graph.comp_param import CompParam
from mizore.utils.type_check import is_number

from jax.numpy import abs
from jax.numpy.linalg import solve
from jax import grad
from mizore import jax_array

class ValVar:
    """
    The class for 'Value with Variance'
    Package for similar function: https://uncertainties.readthedocs.io/en/latest/index.html
    """
    def __init__(self, mean: Union[CompParam, Iterable, numbers.Number, None],
                 var: Union[CompParam, Iterable, numbers.Number] = None, name=None, zero_var=False):
        self.mean: CompParam
        if isinstance(mean, CompParam):
            self.mean = CompParam()
            self.mean.bind_to(mean)
        elif mean is None:
            self.mean = CompParam()
        else:
            try:
                self.mean = CompParam(val=jax_array(mean))
            except Exception:
                raise TypeError()

        self.var: CompParam
        if not zero_var:
            if isinstance(var, CompParam):
                self.var = CompParam()
                self.var.bind_to(var)
            elif var is not None:
                try:
                    self.var = CompParam(val=jax_array(var))
                except Exception:
                    raise TypeError()
            else:
                self.var = CompParam()
        else:
            if var is not None:
                raise Exception()
            # If zero_var is True
            # Make self.var a zero matrix with the same shape as self.val
            self.var = self.mean * 0.0


        self.name = name if name is not None else "Untitled"
        self.mean.name = self.name + "-Mean"
        self.var.name = self.name + "-Var"

    def copy_with_map_dict(self, new_elem_dict):
        if self in new_elem_dict.keys():
            return new_elem_dict[self]
        new_valvar = copy(self)
        new_elem_dict[self] = new_valvar
        new_valvar.var = self.var.copy_with_map_dict(new_elem_dict)
        new_valvar.mean = self.mean.copy_with_map_dict(new_elem_dict)
        return new_valvar

    def set_home_node(self, home_node):
        self.var.home_node = home_node
        self.mean.home_node = home_node

    def replica(self):
        return ValVar(self.mean.replica(), self.var.replica())

    def set_value(self, other: ValVar):
        self.mean.set_value(other.mean)
        self.var.set_value(other.var)

    def bind_to(self, other: ValVar):
        self.mean.bind_to(other.mean)
        self.var.bind_to(other.var)

    @property
    def val(self):
        """
        Alias for self.mean
        """
        return self.mean

    # See https://en.wikipedia.org/wiki/Variance for rules of variance operations
    # Here we are actually calculating the upper bound of the variance
    # because don't know the covariance among the variables

    # Also see https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables

    def __ror__(self, op):
        first_grad = grad(op)
        second_grad = grad(first_grad)
        mean = (op | self.mean) + 0.5 * (second_grad | self.mean) * self.var
        var = ((first_grad | self.mean) ** 2) * self.var - 0.25 * ((second_grad | self.mean) ** 2) * (self.var ** 2)
        return ValVar(mean, var)

    def __mul__(self, other):
        if is_number(other):
            mean = self.mean * other
            # Var[cX] = c^2 Var[X]
            var = self.var*(other*other)
            return ValVar(mean, var)
        elif isinstance(other, ValVar):
            mean = self.mean * other.mean
            # Var[XY] >= E[X]^2 Var[Y] + E[Y]^2 Var[X] + Var[X] Var[Y]]
            var = ((abs | self.mean) ** 2) * other.var + ((abs | other.mean) ** 2) * self.var + other.var * self.var
            return ValVar(mean, var)
        else:
            raise ValVarTypeError(other)

    def __add__(self, other):
        if is_number(other):
            mean = self.mean + other
            # Add a constant won't change the variance
            var = self.var
            return ValVar(mean, var)
        elif isinstance(other, ValVar):
            mean = self.mean + other.mean
            # Var[X+Y] >= Var[X] + Var[Y]
            var = self.var + other.var
            return ValVar(mean, var)
        else:
            raise ValVarTypeError(other)

    def __sub__(self, other):
        if is_number(other):
            mean = self.mean - other
            # Subtract a constant won't change the variance
            var = self.var
            return ValVar(mean, var)
        elif isinstance(other, ValVar):
            mean = self.mean - other.mean
            # Var[X-Y] >= Var[X] + Var[Y]
            var = self.var + other.var
            return ValVar(mean, var)
        else:
            raise ValVarTypeError(other)

    def __truediv__(self, other):
        if is_number(other):
            mean = self.mean / other
            var = self.var / (other * other)
            return ValVar(mean, var)
        elif isinstance(other, ValVar):
            return self * (1/other)
        else:
            raise ValVarTypeError(other)

    def __neg__(self):
        mean = -self.mean
        var = self.var
        return ValVar(mean, var)

    def __rmul__(self, other):
        # Assume Commutative
        return self * other

    def __radd__(self, other):
        # Assume Commutative
        return self + other

    def __rsub__(self, other):
        # Assume Commutative
        return (-self) + other

    def __rtruediv__(self, other):
        if is_number(other):
            mean = other / self.mean + other * (self.mean ** (-3)) * self.var
            var = (self.mean ** (-4)) * self.var - (self.mean ** (-6)) * (self.var ** 2)
            var = (other**2) * var
            return ValVar(mean, var)
        elif isinstance(other, ValVar):
            assert False
        else:
            raise ValVarTypeError(other)

    def __pow__(self, other):
        print("Warning: __pow__ is not tested")

    def __rpow__(self, other):
        raise NotImplementedError()

    def __matmul__(self, other):
        pass

    def inv(self):
        pass

    def conjugate(self):
        return ValVar(self.mean.conjugate(), self.var.conjugate())

    def value(self):
        return self.mean.value(), self.var.value()

    def simple_unary_op(self, op):
        return ValVar(CompParam.unary_operator(self.mean, op), CompParam.unary_operator(self.var, op))

    @classmethod
    def binary_operator(cls, valvar1: ValVar, valvar2: ValVar, op):
        valvar_tuple = ValVar.tuple([valvar1, valvar2])
        return (lambda arg: op(arg[0], arg[1])) | valvar_tuple

    @classmethod
    def tuple(cls, valvars: List[ValVar]):
        means = [valvar.mean for valvar in valvars]
        vars_ = [valvar.var for valvar in valvars]
        means_param = CompParam.tuple(means)
        vars_param = CompParam.tuple(vars_)
        return ValVar(means_param, vars_param)

    @classmethod
    def array(cls, valvars: List[ValVar]):
        means = [valvar.mean for valvar in valvars]
        vars_ = [valvar.var for valvar in valvars]
        means_param = CompParam.array(means)
        vars_param = CompParam.array(vars_)
        return ValVar(means_param, vars_param)

    def get_by_index(self, index):
        return self.simple_unary_op(lambda arg: arg[index])

    def show_value(self):
        print("Mean value:")
        self.mean.show_value()
        print("Variance:")
        self.var.show_value()


class ValVarTypeError(Exception):
    def __init__(self, other):
        msg: str
        if isinstance(other, CompParam):
            msg = "ValVar cannot interact with CompParam directly. \n" + \
                  "If you know what you are doing, you can use ValVar(param, 0) instead."
        else:
            msg = f"ValVar cannot interact with {other.__class__.__name__} directly."
        super().__init__(msg)