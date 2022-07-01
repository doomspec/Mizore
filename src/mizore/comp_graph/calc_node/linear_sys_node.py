from mizore.comp_graph.node.calc_node import CalcNode
from mizore.comp_graph.valvar import ValVar
import numpy as np


class LinearSysNode(CalcNode):
    def __init__(self, A_mat: ValVar, b_vec: ValVar, name=None):
        super().__init__(name=name)
        self.add_output_param("Solution", ValVar(None, name="Solution"))
        self.add_input_param("A_mat", ValVar(None, name="A_mat"))
        self.add_input_param("b_vec", ValVar(None, name="b_vec"))
        self.A_mat.bind_to(A_mat)
        self.b_vec.bind_to(b_vec)

    def calc(self, cache_key=None):
        res = solve_linear_system(self.A_mat, self.b_vec, cache_key=cache_key)
        self.solution.bind_to(res)

    @property
    def A_mat(self) -> ValVar:
        return self.inputs["A_mat"]

    @property
    def b_vec(self)-> ValVar:
        return self.inputs["b_vec"]

    @property
    def solution(self)-> ValVar:
        return self.outputs["Solution"]

    def __call__(self, *args, **kwargs):
        return self.outputs["Solution"]

def get_linear_system_partials(A_mat: ValVar, b_vec: ValVar, cache_key=None):
    A_mean = A_mat.mean.get_value(cache_key=cache_key)
    b_mean = b_vec.mean.get_value(cache_key=cache_key)
    dim = len(b_mean)
    assert len(A_mean) == dim
    A_mean_inv = np.linalg.pinv(A_mean)  # Get the pseudo inverse of A
    x_on_mean = A_mean_inv @ b_mean
    x_partial_A = [[None for _ in range(dim)] for __ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            x_partial_A[i][j] = -A_mean_inv[:, i] * x_on_mean[j]
    x_double_partial_A = [[None for _ in range(dim)] for __ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            x_double_partial_A[i][j] = -2 * A_mean_inv[:, i] * x_partial_A[i][j][j]
    x_partial_b = [A_mean_inv[:, i] for i in range(dim)]
    return x_on_mean, x_partial_A, x_double_partial_A, x_partial_b


def solve_linear_system(A_mat: ValVar, b_vec: ValVar, cache_key=None):
    x_on_mean, x_partial_A, x_double_partial_A, x_partial_b = get_linear_system_partials(A_mat, b_vec,
                                                                                         cache_key=cache_key)
    dim = len(x_partial_b)

    # We will calculate the shift of mean value and variance from the variance from A_mat and b_vec

    A_var = A_mat.var.get_value(cache_key=cache_key)
    b_var = b_vec.var.get_value(cache_key=cache_key)

    mean_shift = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            mean_shift += x_double_partial_A[i][j] / 2 * A_var[i][j]

    x_mean_shifted = x_on_mean + mean_shift
    x_var = np.zeros(dim)

    for i in range(dim):
        for j in range(dim):
            x_var += ((x_partial_A[i][j]) ** 2) * A_var[i][j] - 0.25 * (x_double_partial_A[i][j] ** 2) * (
                        A_var[i][j] ** 2)
    for i in range(dim):
        x_var += ((x_partial_b[i]) ** 2) * b_var[i]

    return ValVar(x_mean_shifted, x_var)
