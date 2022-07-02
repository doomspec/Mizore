from __future__ import annotations

import numbers
from copy import copy
from typing import List, Callable, Set, Dict, Tuple

from jax import numpy

from mizore.comp_graph.computable import Computable
from mizore.utils.type_check import is_number
from mizore import jax_array


class CompParam(Computable):
    def __init__(self, args=None, operator=None, val=None, name=None, home_node=None):
        self.name = name
        # The node that the parameter is based.
        # Can be None when the parameter does not directly depend on a node
        self.home_node = home_node
        # The list of the parameters that this parameter depends on
        # Should be None when home_node is not None
        self.args: List[CompParam] = args
        self.operator: Callable

        if args is None:
            self.args = []
            if val is not None:
                self.operator = lambda: val
                assert operator is None
            else:
                self.operator = operator
        else:
            self.args = args
            self.operator = operator

        self.is_variable = False
        self.cache_val = None

    def set_home_node(self, home_node):
        self.home_node = home_node

    def show_value(self):
        print(f"{self.name if self.name is not None else 'Untitled'}: {self.value()}")

    def eval(self):
        """
        Alias of self.value()
        """
        return self.value()

    def value(self):
        return self.get_value()

    def del_cache(self):
        self.cache_val = None

    """
    def del_cache_recursive(self):
        CompParam.del_cache_recursive_(self, set())
    
    @classmethod
    def del_cache_recursive_(cls, param: CompParam, touched_param: Set[CompParam]):
        param.del_cache()
        for arg_param in param.args:
            if arg_param not in touched_param:
                touched_param.add(arg_param)
                CompParam.del_cache_recursive_(arg_param, touched_param)
    """

    def eval_and_cache(self):
        return self.get_value()

    def get_value(self):
        return CompParam._get_value(self)

    @classmethod
    def _get_value(cls, param: CompParam):
        if param.cache_val is not None:
            return param.cache_val
        if hasattr(param.home_node, "calc"):
            param.home_node.calc()
        if len(param.args) != 0:
            arg_vals = []
            for arg in param.args:
                arg_vals.append(CompParam._get_value(arg))
            val = param.operator(*arg_vals)
        else:
            if param.operator is not None:
                val = param.operator()
            else:
                raise NotComputedError(param)

        param.cache_val = val
        return val

    def get_eval_fun(self) -> Tuple[Callable, List[CompParam], List[numbers.Number]]:
        eval_fun, var_list, var_dict = CompParam._get_eval_fun(self)
        init_val = numpy.array([var.value() for var in var_list])
        return lambda args: eval_fun(*args), var_list, init_val

    @classmethod
    def _get_eval_fun(cls, param: CompParam):
        """
        :return: A tuple of three elements
        eval_fun: the function for evaluation, whose variable is specified by var_list and var_dict
        var_list: the list of variables for the eval_fun. The order matters
        var_dict: the dict that map variable (CompParam) to its position in var_list.
        var_dict should be maintained to be consistent with var_list
        """
        if param.operator is None:
            if hasattr(param.home_node, "calc"):
                param.home_node.calc()
                if param.operator is None:
                    raise NotComputedError(param)
            else:
                raise NotComputedError(param)
        if param.is_variable:
            return lambda x: x, [param], {param: 0}
        n_child = len(param.args)
        if n_child == 0:
            return lambda: param.operator(), [], dict()
        elif n_child == 1:
            sub_eval, sub_vars, sub_vars_dict = CompParam._get_eval_fun(param.args[0])
            return lambda *args: param.operator(sub_eval(*args)), sub_vars, sub_vars_dict
        elif n_child == -1:  # == 2
            eval0, vars0, var_dict0 = CompParam._get_eval_fun(param.args[0])
            eval1, vars1, var_dict1 = CompParam._get_eval_fun(param.args[1])
            # TODO this part is not thoroughly tested! I guess it is correct from simple test cases.
            # The result of both branches should be the same
            if len(vars0) < len(vars1) + 5:  # If vars0 is smaller than vars1  # 5 is an ad hoc value
                return CompParam.get_two_arg_eval_fun(param.operator, eval0, vars0, var_dict0, eval1, vars1,
                                                      var_dict1)
            else:
                return CompParam.get_two_arg_eval_fun(
                    lambda *args: param.operator(*args[len(vars0):], *args[:len(vars0)]),
                    eval1, vars1, var_dict1, eval0, vars0, var_dict0)
        else:
            eval_list = []
            vars_list = []
            var_dict_list = []
            for i in range(n_child):
                _eval, _vars, var_dict = CompParam._get_eval_fun(param.args[i])
                eval_list.append(_eval)
                vars_list.append(_vars)
                var_dict_list.append(var_dict)
            return CompParam.get_multi_arg_eval_fun(param.operator, eval_list, vars_list, var_dict_list)

    @classmethod
    def get_two_arg_eval_fun(cls, op, eval0, vars0, var_dict0: Dict, eval1, vars1, var_dict1: Dict):
        overlap = key_overlap(var_dict0, var_dict1)
        len0 = len(vars0)
        new_var_dict = {key: value for key, value in var_dict0.items()}
        new_var_list = [var for var in vars0]
        value_map = list(range(len0))  # order of vars in vars0 will remain the same
        var_index = len0  # (var_index + len0) is the new indices of the different variables in vars1
        for var in vars1:
            if var not in overlap:
                value_map.append(var_index)
                new_var_dict[var] = var_index
                new_var_list.append(var)
                var_index += 1
            else:
                value_map.append(var_dict0[var])  # Map the repeated variable to its position in vars0

        def new_operator(*args):
            expanded_args = [args[i] for i in value_map]  # I tested this. This should work under jax
            return op(eval0(*expanded_args[:len0]), eval1(*expanded_args[len0:]))

        return new_operator, new_var_list, new_var_dict

    @classmethod
    def get_multi_arg_eval_fun(cls, op, eval_list: List, vars_list: List, var_dict_list: List[Dict]):
        var_added = set(vars_list[0])
        value_map = list(range(len(vars_list[0])))
        new_var_dict = {key: value for key, value in var_dict_list[0].items()}
        new_var_list = [var for var in vars_list[0]]
        var_index = len(vars_list[0])
        for i in range(1, len(eval_list)):
            for var in vars_list[i]:
                if var not in var_added:
                    value_map.append(var_index)
                    new_var_list.append(var)
                    var_added.add(var)
                    new_var_dict[var] = var_index
                    var_index += 1
                else:
                    value_map.append(new_var_dict[var])
        delimiters = [0] * (len(vars_list) + 1)
        for i in range(len(vars_list)):
            delimiters[i + 1] = delimiters[i] + len(vars_list[i])
        n_eval = len(eval_list)

        def new_operator(*args):
            expanded_args = [args[i] for i in value_map]
            eval_res = [eval_list[i](*expanded_args[delimiters[i]:delimiters[i + 1]]) for i in range(n_eval)]
            return op(*eval_res)

        return new_operator, new_var_list, new_var_dict

    def replica(self):
        return CompParam(args=self.args, operator=self.operator)

    def set_to_variable(self):
        self.is_variable = True

    def change_to_const(self, val=None):
        self.is_variable = False
        if val is None:
            self.set_value(self.value())
        else:
            if isinstance(val, CompParam):
                self.set_value(val.value())
            elif is_number(val):
                self.set_value(val)
            else:
                raise TypeError()

    def bind_to(self, param: CompParam):
        if isinstance(param, CompParam):
            self.operator = lambda x: x
            self.args = [param]
            self.is_variable = False
        else:
            raise TypeError()

    def set_value(self, val):
        if is_number(val):
            self.operator = lambda: val
        elif isinstance(val, CompParam):
            self.operator = val.operator
            self.args = val.args
        else:
            raise TypeError()

    @classmethod
    def unary_operator(cls, param, op):
        return CompParam(args=[param],
                         operator=lambda *args: op(*args))

    @classmethod
    def binary_operator(cls, left, right, op):
        if is_number(right):
            if isinstance(left, CompParam):
                return cls.unary_operator(param=left, op=lambda E: op(E, right))
            else:
                raise Exception(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        elif is_number(left):
            if isinstance(right, CompParam):
                return cls.unary_operator(param=right, op=lambda E: op(left, E))
            else:
                raise Exception(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        else:
            return CompParam(args=[left, right], operator=lambda arg0, arg1: op(arg0, arg1))

    def left_apply(self, op, right):
        if is_number(right):
            t = lambda v: op(v, right)
            new = self.unary_operator(param=self, op=t)
        elif isinstance(right, CompParam):
            new = self.binary_operator(left=self, right=right, op=op)
        else:
            raise TypeError(f"Type {right.__class__.__name__} is not supported to operate with CompParam")
        return new

    def right_apply(self, op, left):
        if is_number(left):
            t = lambda v: op(left, v)
            new = self.unary_operator(param=self, op=t)
        elif isinstance(left, CompParam):
            new = self.binary_operator(left=left, right=self, op=op)
        else:
            raise TypeError(f"Type {left.__class__.__name__} is not supported to operate with CompParam")
        return new

    def __mul__(self, other):
        return self.left_apply(numpy.multiply, other)

    def __add__(self, other):
        return self.left_apply(numpy.add, other)

    def __sub__(self, other):
        return self.left_apply(numpy.subtract, other)

    def __truediv__(self, other):
        return self.left_apply(numpy.true_divide, other)

    def __neg__(self):
        return self.unary_operator(param=self, op=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self.left_apply(numpy.float_power, other)

    def __rpow__(self, other):
        return self.right_apply(numpy.float_power, other)

    def __rmul__(self, other):
        return self.right_apply(numpy.multiply, other)

    def __radd__(self, other):
        return self.right_apply(numpy.add, other)

    def __rsub__(self, other):
        return self.right_apply(numpy.subtract, other)

    def __rtruediv__(self, other):
        return self.right_apply(numpy.true_divide, other)

    def __ror__(self, op):
        return CompParam.unary_operator(self, op)

    def __str__(self):
        res = [f"Class:{self.__class__.__name__}\nName:{self.name}\nHome:{self.home_node}\n", f"Args: {self.args}\n",
               f"Operator: {self.operator}"]
        return "".join(res)

    def conjugate(self):
        return CompParam.unary_operator(self, numpy.conjugate)

    def copy_with_map_dict(self, new_elem_dict):
        # TODO test this
        if self in new_elem_dict.keys():
            return new_elem_dict[self]
        new_param = copy(self)
        new_elem_dict[self] = new_param
        new_param.args = [arg.copy_with_map_dict(new_elem_dict) for arg in self.args]
        new_param.home_node = self.home_node.copy_with_map_dict(new_elem_dict)
        return new_param

    @classmethod
    def tuple(cls, params: List[CompParam]):
        tuple_param = CompParam(params, operator=(lambda *arg: tuple(arg)))
        return tuple_param

    @classmethod
    def array(cls, params: List[CompParam]):
        array_param = CompParam(params, operator=(lambda *arg: jax_array(arg)))
        return array_param

    def get_by_index(self, index):
        return CompParam.unary_operator(self, lambda arg: arg[index])


def key_overlap(dict1: Dict, dict2: Dict) -> Set:
    if len(dict1) > len(dict2):
        return key_overlap(dict2, dict1)
    overlap = set()
    dict2_keys = dict2.keys()
    for key in dict1.keys():
        if key in dict2_keys:
            overlap.add(key)
    return overlap


class NotComputedError(Exception):
    def __init__(self, param: CompParam):
        msg: str
        self.param = param
        if param.home_node is not None:
            msg = f"\nThe following CompParam does not have operator.\n{param} " + \
                  f"\nDid you transpile its home node {param.home_node.name}?"
        else:
            msg = f"\nThe following CompParam does not have operator.\n{param} " + \
                  f"\nThere is likely a bug in the code."
        super().__init__(msg)
