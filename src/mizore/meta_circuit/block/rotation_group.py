from __future__ import annotations

from typing import List

from mizore.backend_circuit.rotations import PauliRotation

from mizore.operators import QubitOperator

from .block import Block

from mizore import np_array
from jax.numpy import pi


class RotationGroup(Block):
    def __init__(self, qubit_operator: QubitOperator = None, qset_ops_weight: List = None, fixed_angle_shift=None):
        self.qset_ops_weights: List
        if qset_ops_weight is not None:
            self.qset_ops_weights = qset_ops_weight
        else:
            self.qset_ops_weights = []
            for term in qubit_operator.qset_ops_weight():
                if len(term[0]) != 0:
                    self.qset_ops_weights.append(term)
        Block.__init__(self, len(self.qset_ops_weights), fixed_param=fixed_angle_shift)

    def get_gates(self, params):
        return list(map(get_pauli_rotation, self.qset_ops_weights, self._fixed_param + params))

    def get_inverse_block(self) -> RotationGroup:
        new_qset_ops_weights = [(qset, ops, -weight) for qset, ops, weight in self.qset_ops_weights]
        return RotationGroup(qset_ops_weight=new_qset_ops_weights)

    def get_gradient_blocks(self, param_index: int, params=None):
        angle_shift1 = np_array([0.0] * self.n_param)
        angle_shift1[param_index] += pi / 2
        angle_shift2 = np_array([0.0] * self.n_param)
        angle_shift2[param_index] -= pi / 2
        return [(0.5, RotationGroup(qset_ops_weight=self.qset_ops_weights, fixed_angle_shift=angle_shift1+self.fixed_param)),
                (-0.5, RotationGroup(qset_ops_weight=self.qset_ops_weights, fixed_angle_shift=angle_shift2+self.fixed_param))]


def get_pauli_rotation(pauli_term, angle):
    qset, ops, weight = pauli_term
    return PauliRotation(qset, ops, weight * angle)
