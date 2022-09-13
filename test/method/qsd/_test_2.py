from chemistry.simple_mols import simple_8_qubit_h4
from method.qsd.test_variance import make_krylov_single_ref_random_mat
from mizore.comp_graph.value import Value
from mizore.method.qsd.generalized_eigv import generalized_eigv_by_wang
from mizore.method.qsd.krylov_single_ref_classical import quantum_krylov_single_ref_classical
import numpy as np
hamil, init_circuit = simple_8_qubit_h4()

n_basis = 4
delta = 1.0
realistic = True
eigv_classical, H_classical, S_classical = quantum_krylov_single_ref_classical(init_circuit, hamil, n_basis, delta,
                                                                               get_H_S=True)
print(eigv_classical)

var = 0.000001#1e-14

H_mat = make_krylov_single_ref_random_mat(H_classical, var)
S_mat = make_krylov_single_ref_random_mat(S_classical, 0.0)

def generalized_eigvals(H_, S_):
    return generalized_eigv_by_wang(H_, S_, eigvals_only=True, eps=1e-12)


res = Value.binary_operator(H_mat, S_mat, generalized_eigvals)



res.show_value()
res.var.show_value()

print(np.linalg.cond(H_classical))
print(np.linalg.cond(S_classical))