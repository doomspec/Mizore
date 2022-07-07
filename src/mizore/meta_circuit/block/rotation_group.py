from __future__ import annotations

from typing import List

from mizore.backend_circuit.rotations import PauliRotation

from mizore.operators import QubitOperator

from .block import Block

from mizore import np_array
from jax.numpy import pi


class RotationGroup(Block):
    """
    Implement exp(iP_0t/2)exp(iP_1t/2)exp(iP_2t/2)...
    """

    def __init__(self, qubit_operator: QubitOperator = None, qset_op_weight: List = None, fixed_angle_shift=None):
        self.qset_op_weights: List
        if qset_op_weight is not None:
            self.qset_op_weights = qset_op_weight
        else:
            self.qset_op_weights = []
            for term in qubit_operator.qset_op_weight():
                if len(term[0]) != 0:
                    self.qset_op_weights.append(term)
        Block.__init__(self, len(self.qset_op_weights), fixed_param=fixed_angle_shift)

    def get_gates(self, params):
        return list(map(get_pauli_rotation, self.qset_op_weights, self._fixed_param + params))

    def get_inverse_block(self) -> RotationGroup:
        new_qset_op_weights = [(qset, ops, -weight) for qset, ops, weight in self.qset_op_weights]
        return RotationGroup(qset_op_weight=new_qset_op_weights, fixed_angle_shift=self.fixed_param)

    def get_gradient_blocks(self, param_index: int, params=None):
        this_weight = self.qset_op_weights[param_index][2]
        angle_shift1 = np_array([0.0] * self.n_param)
        angle_shift1[param_index] += pi / (2*this_weight)
        angle_shift2 = np_array([0.0] * self.n_param)
        angle_shift2[param_index] -= pi / (2*this_weight)
        return [(0.5*this_weight, RotationGroup(qset_op_weight=self.qset_op_weights, fixed_angle_shift=angle_shift1+self.fixed_param)),
                (-0.5*this_weight, RotationGroup(qset_op_weight=self.qset_op_weights, fixed_angle_shift=angle_shift2+self.fixed_param))]

def get_pauli_rotation(pauli_term, angle):
    qset, ops, weight = pauli_term
    return PauliRotation(qset, ops, weight * angle)
