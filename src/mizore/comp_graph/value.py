from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mizore.comp_graph.comp_node import CompNode

import numbers
import time
from copy import copy
from typing import List, Callable, Set, Dict, Tuple, Union

from jax import jacfwd, jit
import jax.numpy as jnp
import numpy as np
from jax.numpy import complex128
from numpy.random import default_rng

from mizore.utils.type_check import is_number
from mizore import jax_array, to_jax_array
from mizore.comp_graph.comp_graph import CompGraph


class Value:
    def __init__(self, val=None, args=None, operator=None, name=None, home_node=None):
        self.name = name
        # The node that the parameter is based.
        # Can be None when the parameter does not directly depend on a node
        self.home_node: CompNode = home_node
        # The list of the parameters that this parameter depends on
        # Should be None when home_node is not None
        self.args: List[Value] = args
        self.operator: Callable

        if args is None:
            self.args = []
            if val is not None:
                assert operator is None
                if isinstance(val, Value):
                    self.operator = lambda x: x
                    self.args = [val]
                else:
                    self.operator = lambda: to_jax_array(val)
            else:
                # Here, operator might be None. But it's okay. We just initiate an empty Value.
                self.operator = operator
        else:
            self.args = args
            self.operator = operator

        self.is_parameter = False
        self.cache_val = None

        self.is_indep_random = False
        self.var_constructed: Union[Value, None] = None
        self.var_dependency_cache = None

        self.first_var_cache = None
        self.second_var_cache = None
        self.shifted_mean_cache = None

        self.linear_approx: bool = False
        self.const_approx: bool = False
        self.approx_args = []
        self.approx_fun = None

    @property
    def var(self) -> Value:
        if self.is_indep_random:
            return self.var_constructed
        else:
            if self.first_var_cache is not None:
                return self.first_var_cache
            var = self._get_var(order=1)
            self.first_var_cache = var
            return var

    @property
    def var_second_order(self) -> Value:
        if self.is_indep_random:
            return self.var_constructed
        else:
            if self.second_var_cache is not None:
                return self.second_var_cache
            var = self._get_var(order=2)
            self.second_var_cache = var
            return var

    @property
    def mean_second_order(self):
        return self.get_shifted_mean()

    def get_approx_eval_on_var(self):
        approx_eval, var_list, var_dict = Value._get_approx_eval_on_var(self)
        init_val = [var.value() for var in var_list]
        return lambda args: approx_eval(*args), var_list, init_val

    def generate_linear_approx_on_var(self):
        self.linear_approx = False
        eval_on_var, var_list, init_val = self.get_approx_eval_on_var()
        self.linear_approx = True
        self.approx_args = var_list
        init_func = eval_on_var(init_val)

        if len(var_list) == 0:
            self.approx_fun = lambda: init_func
            self.approx_args = []
            return

        grads = jacfwd(eval_on_var)(init_val)

        def approx_fun(*args):
            res = init_func
            for i in range(len(args)):
                res += jnp.tensordot(grads[i], (args[i] - init_val[i]), jnp.ndim(init_val[i]))
            return res

        self.approx_fun = approx_fun

    def generate_const_approx_on_var(self):
        self.const_approx = False
        eval_on_var, var_list, init_val = self.get_approx_eval_on_var()
        self.const_approx = True
        init_func = eval_on_var(init_val)

        if len(var_list) == 0:
            self.approx_fun = lambda: init_func
            self.approx_args = []
            return

        grads = jacfwd(eval_on_var)(init_val)
        std_deviation_list = [jnp.sqrt(var.var.value()) for var in var_list]

        var = init_func * 0.0
        for i in range(len(grads)):
            var += jnp.tensordot(grads[i] ** 2, std_deviation_list[i] ** 2, jnp.ndim(init_val[i]))

        const_variable = Variable(eval_on_var(init_val), var)

        self.approx_fun = lambda x: x
        self.approx_args = [const_variable]

    @classmethod
    def _get_approx_eval_on_var(cls, param: Value) -> Tuple[Callable, List[Value], Dict[Value, int]]:
        if param.is_indep_random:
            return lambda x: x, [param], {param: 0}

        if param.linear_approx:
            if param.approx_fun is None:
                param.generate_linear_approx_on_var()
            return lambda *args: param.approx_fun(*args), param.approx_args, {param.approx_args[i]: i for i in
                                                                              range(len(param.approx_args))}
        if param.const_approx:
            if param.approx_fun is None:
                param.generate_const_approx_on_var()
            return lambda *args: param.approx_fun(*args), param.approx_args, {param.approx_args[i]: i for i in
                                                                              range(len(param.approx_args))}

        if param.operator is None:
            if hasattr(param.home_node, "calc"):
                param.home_node.calc()
                if param.operator is None:
                    raise NotComputedError(param)
            else:
                raise NotComputedError(param)
        n_child = len(param.args)
        if n_child == 0:
            return lambda: param.operator(), [], dict()
        elif n_child == 1:
            sub_eval, sub_vars, sub_vars_dict = Value._get_approx_eval_on_var(param.args[0])
            return lambda *args: param.operator(sub_eval(*args)), sub_vars, sub_vars_dict
        elif n_child > 1:
            eval_list = []
            vars_list = []
            var_dict_list = []
            for i in range(n_child):
                _eval, _vars, var_dict = Value._get_approx_eval_on_var(param.args[i])
                eval_list.append(_eval)
                vars_list.append(_vars)
                var_dict_list.append(var_dict)
            return Value.get_multi_arg_eval_fun(param.operator, eval_list, vars_list, var_dict_list)

    def _get_var(self, order=1) -> Value:
        eval_func, var_list, init_list = self.get_approx_eval_on_var()

        if len(var_list) == 0:
            return Value(0.0)  # TODO: does shape matter?

        first_grad = jacfwd(eval_func)

        def variance_from_first_grad(variable, variable_var):
            grads = first_grad(variable)
            to_sum = [multiply_and_sum(grads[i] ** 2, variable_var[i]) for i in range(len(grads))]
            return sum(to_sum)

        variable_vars = Value(args=[var.var_constructed for var in var_list], operator=lambda *args: list(args))
        variables = Value(args=var_list, operator=lambda *args: list(args))
        first_grad_contribution = Value.binary_operator(variables, variable_vars, variance_from_first_grad)

        if order == 1:
            return first_grad_contribution

        assert False  # I cannot ensure the correctness of the following

        second_grad = jacfwd(first_grad)

        def variance_from_second_grad(variable, variable_var):
            second_grads = second_grad(variable)
            contracted0 = [sum([multiply_and_sum(second_grads[i][j] ** 2, variable_var[j])
                                for j in range(len(second_grads[i]))])
                           for i in range(len(second_grads))]
            contracted1 = [multiply_and_sum(contracted0[i], variable_var[i]) for i in range(len(second_grads))]
            summed = -0.25 * sum(contracted1)
            return summed

        second_grad_contribution = Value.binary_operator(variables, variable_vars, variance_from_second_grad)

        """
        for i in range(len(var_list)):
            second_grad = jacfwd(first_grad_func_list[i], argnums=i)
            second_grad_contributions.append(
                Value.binary_operator(Value.binary_operator(-0.25 * (Value(args=var_list, operator=second_grad) ** 2),
                                                            var_list[i].var_constructed, multiply_and_sum),
                                      var_list[i].var_constructed, multiply_and_sum))
        """
        # second_grad_contribution = Value(args=[Value.array(second_grad_contributions)], operator=sum_first_axis)

        if order == 2:
            return first_grad_contribution + second_grad_contribution
        else:
            raise Exception()

    def get_shifted_mean(self):
        eval_func, var_list, _ = self.get_eval_on_var()
        second_grad_contributions = []
        for i in range(len(var_list)):
            second_grad = jacfwd(jacfwd(eval_func, argnums=i), argnums=i)
            second_grad_contributions.append(Value.binary_operator(0.5 * Value(args=var_list, operator=second_grad),
                                                                   var_list[i].var_constructed, multiply_and_sum))
        return self + Value(args=[Value.array(second_grad_contributions)], operator=sum_first_axis)

    @classmethod
    def random_variable(cls, mean, var) -> Value:
        param = Value(val=mean)
        param.set_to_random_variable(var, check_valid=False)
        return param

    def set_to_random_variable(self, var=None, check_valid=True):
        if check_valid:
            # Check whether the Value is already dependent on other indep random variable
            _, var_list, _ = self.get_eval_on_var()
            assert len(var_list) == 0
        self.is_indep_random = True
        self.var_constructed = Value(val=var)

    def set_to_not_random(self):
        self.is_indep_random = False
        self.var_constructed = None

    def set_home_node(self, home_node):
        self.home_node = home_node

    def show_value(self):
        print(f"{self.name if self.name is not None else 'Untitled'}: {self.value()}")

    def show_variance(self):
        print(f"{self.name if self.name is not None else 'Untitled'}-Var: {self.var.value()}")

    def show_std_devi(self):
        """
        Show the standard deviation of the value
        """
        print(f"{self.name if self.name is not None else 'Untitled'}-StdDevi: {jnp.sqrt(self.var.value())}")

    def del_cache(self):
        self.cache_val = None
        self.var_dependency_cache = None
        self.first_var_cache = None
        self.second_var_cache = None
        self.shifted_mean_cache = None

        self.approx_args = []
        self.approx_fun = None

    def del_cache_recursive(self):
        Value._del_cache_recursive(self, set())

    @classmethod
    def _del_cache_recursive(cls, param: Value, touched_param: Set[Value]):
        param.del_cache()
        for arg_param in param.args:
            if arg_param not in touched_param:
                touched_param.add(arg_param)
                Value._del_cache_recursive(arg_param, touched_param)

    def value(self):
        return self.get_value()

    def get_value(self):
        return Value._get_value(self)

    @classmethod
    def _get_value(cls, param: Value):
        if param.cache_val is not None:
            return param.cache_val
        if hasattr(param.home_node, "calc"):
            param.home_node.calc()
        if len(param.args) != 0:
            arg_vals = []
            for arg in param.args:
                arg_vals.append(Value._get_value(arg))
            val = param.operator(*arg_vals)
        else:
            if param.operator is not None:
                val = param.operator()
            else:
                raise NotComputedError(param)

        param.cache_val = val
        return val

    def get_eval_on_var(self) -> Tuple[Callable, List[Value], List[numbers.Number]]:
        eval_fun, var_list, var_dict = Value._get_eval_fun(self, (lambda param: param.is_indep_random))
        init_val = [var.value() for var in var_list]
        return lambda args: eval_fun(*args), var_list, init_val

    def get_eval_on_param(self) -> Tuple[Callable, List[Value], List[numbers.Number]]:
        eval_fun, var_list, var_dict = Value._get_eval_fun(self, (lambda param: param.is_parameter))
        init_val = [var.value() for var in var_list]
        return lambda args: eval_fun(*args), var_list, init_val

    @classmethod
    def _get_eval_fun(cls, param: Value, terminal_checker) -> Tuple[Callable, List[Value], Dict[Value, int]]:
        """
        :return: A tuple of three elements
        eval_fun: the function for evaluation, whose variable is specified by var_list and var_dict
        var_list: the list of variables for the eval_fun. The order matters
        var_dict: the dict that map variable (Value) to its position in var_list.
        var_dict should be maintained to be consistent with var_list
        """
        if param.operator is None:
            if hasattr(param.home_node, "calc"):
                param.home_node.calc()
                if param.operator is None:
                    raise NotComputedError(param)
            else:
                if terminal_checker(param):
                    return lambda x: x, [param], {param: 0}
                else:
                    raise NotComputedError(param)
        if terminal_checker(param):
            return lambda x: x, [param], {param: 0}
        n_child = len(param.args)
        if n_child == 0:
            return lambda: param.operator(), [], dict()
        elif n_child == 1:
            sub_eval, sub_vars, sub_vars_dict = Value._get_eval_fun(param.args[0], terminal_checker)
            return lambda *args: param.operator(sub_eval(*args)), sub_vars, sub_vars_dict
        elif n_child > 1:
            eval_list = []
            vars_list = []
            var_dict_list = []
            for i in range(n_child):
                _eval, _vars, var_dict = Value._get_eval_fun(param.args[i], terminal_checker)
                eval_list.append(_eval)
                vars_list.append(_vars)
                var_dict_list.append(var_dict)
            return Value.get_multi_arg_eval_fun(param.operator, eval_list, vars_list, var_dict_list)
        elif n_child == -1:  # == 2
            eval0, vars0, var_dict0 = Value._get_eval_fun(param.args[0], terminal_checker)
            eval1, vars1, var_dict1 = Value._get_eval_fun(param.args[1], terminal_checker)
            # I don't know whether a special branch for n_child = 2 can increase the efficiency
            # The result of both branches should be the same
            if len(vars0) < len(vars1) + 5:  # If vars0 is smaller than vars1  # 5 is an ad hoc value
                return Value.get_two_arg_eval_fun(param.operator, eval0, vars0, var_dict0, eval1, vars1,
                                                  var_dict1)
            else:
                return Value.get_two_arg_eval_fun(
                    lambda *args: param.operator(*args[len(vars0):], *args[:len(vars0)]),
                    eval1, vars1, var_dict1, eval0, vars0, var_dict0)

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
    def get_multi_arg_eval_fun(cls, op, eval_list: List, vars_list: List[List[Value]], var_dict_list: List[Dict]):
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
            expanded_args = [args[i_] for i_ in value_map]
            eval_res = [eval_list[i_](*expanded_args[delimiters[i_]:delimiters[i_ + 1]]) for i_ in range(n_eval)]
            return op(*eval_res)

        return new_operator, new_var_list, new_var_dict

    def replica(self):
        return Value(args=self.args, operator=self.operator)

    def set_to_variable(self):
        self.is_parameter = True

    def change_to_const(self, val=None):
        self.is_parameter = False
        if val is None:
            self.set_value(self.value())
        else:
            if isinstance(val, Value):
                self.set_value(val.value())
            elif is_number(val):
                self.set_value(val)
            else:
                raise TypeError()

    def bind_to(self, param: Value):
        if isinstance(param, Value):
            self.operator = lambda x: x
            self.args = [param]
            self.is_parameter = False
            self.del_cache()
        else:
            raise TypeError()

    def set_value(self, val: Union[numbers.Number, Value, List[numbers.Number]]):
        self.del_cache()
        if is_number(val):
            self.operator = lambda: val
            self.args = []
        elif isinstance(val, Value):
            self.operator = val.operator
            self.args = val.args
        elif isinstance(val, List):
            self.operator = lambda: jax_array(val)
            self.args = []
        else:
            raise TypeError()

    @classmethod
    def unary_operator(cls, param, op):
        return Value(args=[param],
                     operator=lambda *args: op(*args))

    @classmethod
    def binary_operator(cls, left, right, op):
        if is_number(right):
            if isinstance(left, Value):
                return cls.unary_operator(param=left, op=lambda E: op(E, right))
            else:
                raise Exception(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        elif is_number(left):
            if isinstance(right, Value):
                return cls.unary_operator(param=right, op=lambda E: op(left, E))
            else:
                raise Exception(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        else:
            return Value(args=[left, right], operator=lambda arg0, arg1: op(arg0, arg1))

    def left_apply(self, op, right):
        if is_number(right):
            t = lambda v: op(v, right)
            new = self.unary_operator(param=self, op=t)
        elif isinstance(right, Value):
            new = self.binary_operator(left=self, right=right, op=op)
        else:
            raise TypeError(f"Type {right.__class__.__name__} is not supported to operate with Value")
        return new

    def right_apply(self, op, left):
        if is_number(left):
            t = lambda v: op(left, v)
            new = self.unary_operator(param=self, op=t)
        elif isinstance(left, Value):
            new = self.binary_operator(left=left, right=self, op=op)
        else:
            raise TypeError(f"Type {left.__class__.__name__} is not supported to operate with Value")
        return new

    def __mul__(self, other):
        return self.left_apply(jnp.multiply, other)

    def __add__(self, other):
        return self.left_apply(jnp.add, other)

    def __sub__(self, other):
        return self.left_apply(jnp.subtract, other)

    def __truediv__(self, other):
        return self.left_apply(jnp.true_divide, other)

    def __neg__(self):
        return self.unary_operator(param=self, op=lambda v: jnp.multiply(v, -1))

    def __pow__(self, other):
        return self.left_apply(jnp.float_power, other)

    def __rpow__(self, other):
        return self.right_apply(jnp.float_power, other)

    def __rmul__(self, other):
        return self.right_apply(jnp.multiply, other)

    def __radd__(self, other):
        return self.right_apply(jnp.add, other)

    def __rsub__(self, other):
        return self.right_apply(jnp.subtract, other)

    def __rtruediv__(self, other):
        return self.right_apply(jnp.true_divide, other)

    def __ror__(self, op):
        return Value.unary_operator(self, op)

    def __str__(self):
        res = [f"Class:{self.__class__.__name__}\nName:{self.name}\nHome:{self.home_node}\n", f"Args: {self.args}\n",
               f"Operator: {self.operator}"]
        return "".join(res)

    def conjugate(self):
        return Value.unary_operator(self, jnp.conjugate)

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
    def tuple(cls, params: List[Value]):
        tuple_param = Value(args=params, operator=(lambda *arg: tuple(arg)))
        return tuple_param

    @classmethod
    def array(cls, params: List[Union[Value, List]]):
        array_param = Value(args=params, operator=(lambda *arg: jax_array(arg)))
        return array_param

    @classmethod
    def matrix(cls, param_mat: List[List[Value]]):
        value_mat = [None]*len(param_mat)
        for i in range(len(param_mat)):
            value_mat[i] = Value.array(param_mat[i])
        return Value.array(value_mat)

    def get_by_index(self, index):
        return Value.unary_operator(self, lambda arg: arg[index])

    sample_seed_shift = 0

    """
    def sample_gaussian(self, seed=None):
        rng = default_rng(time.time() * 5 + Value.sample_seed_shift * 11)
        Value.sample_seed_shift += 1
        sqrt_var_mat = np.sqrt(self.var.value())
        mean_mat = self.value()
        if mean_mat.dtype == complex128:
            return rng.normal(mean_mat.real, sqrt_var_mat.real) + rng.normal(mean_mat.imag, sqrt_var_mat.imag)*1j
        else:
            return rng.normal(mean_mat, sqrt_var_mat)
    """

    def gaussian_sample_iter(self, n_sample, use_jit=False):
        eval_func, var_list, init_val = self.get_eval_on_var()
        if use_jit:
            eval_func = jit(eval_func)
        std_devi = [jnp.sqrt(var.var.value()) for var in var_list]
        n_var = len(std_devi)
        for i in range(n_sample):
            sample_variable_val = []
            for var_i in range(n_var):
                rng = default_rng(int(time.time() * 53) + Value.sample_seed_shift * 11 + var_i * 7)
                Value.sample_seed_shift += 1
                sampled_gaussian = rng.normal(init_val[var_i], std_devi[var_i])
                sample_variable_val.append(sampled_gaussian)
            yield eval_func(sample_variable_val)

    def build_graph(self, other_val: Union[List[Value], None] = None):
        if other_val is None:
            return CompGraph([self])
        else:
            return CompGraph([self] + other_val)


def sum_first_axis(arr):
    return jnp.sum(arr, axis=0)


def multiply_and_sum(grad, var):
    # res = jnp.sum(grad * var, axis=tuple(i for i in range(grad.ndim-var.ndim, grad.ndim)))
    res = jnp.tensordot(grad, var, axes=jnp.ndim(var))
    return res


def key_overlap(dict1: Dict, dict2: Dict) -> Set:
    if len(dict1) > len(dict2):
        return key_overlap(dict2, dict1)
    overlap = set()
    dict2_keys = dict2.keys()
    for key in dict1.keys():
        if key in dict2_keys:
            overlap.add(key)
    return overlap


def Variable(mean, var, name=None):
    val = Value(name=name)
    val.set_value(mean)
    val.set_to_random_variable(var, check_valid=False)
    return val


class NotComputedError(Exception):
    def __init__(self, param: Value):
        msg: str
        self.param = param
        if param.home_node is not None:
            msg = f"\nThe following Value does not have operator.\n{param} " + \
                  f"\nDid you transpile its home node {param.home_node.name}?"
        else:
            msg = f"\nThe following Value does not have operator.\n{param} " + \
                  f"\nThere is likely a bug in the code."
        super().__init__(msg)
