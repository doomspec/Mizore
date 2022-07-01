

import math

from mizore.comp_graph.comp_param import CompParam
from mizore.comp_graph.valvar import ValVar
from mizore.comp_graph.var_param import VariableParam

u0 = CompParam()
x0 = VariableParam(1.0)
x1 = VariableParam(2.0)
x2 = VariableParam(3.0)
x3 = VariableParam(4.0)
x4 = VariableParam(5.0)

y0 = ValVar(1.0, 0.1)
y1 = ValVar(0.5, 0.1)

z0 = y0/y1
z1 = y0*(((lambda x: 1/x) | y1))

print(z0.value())
print(z1.value())