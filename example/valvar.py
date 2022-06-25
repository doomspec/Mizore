from mizore.comp_graph.valvar import ValVar
from mizore.comp_graph.var_param import VariableParam

x0 = VariableParam(3.0, name="Var1")
x1 = VariableParam(2.0, name="Var2")

y1 = ValVar(x0, 0.1)
y2 = ValVar(x1, 0.2)

z = (y1+y2)*y1

func, var_list, init_val = z.mean.get_eval_fun()

from jax import grad

print(func(init_val))
print(grad(func)(init_val))