from __future__ import annotations

from mizore.backend_circuit.rotations import PauliRotation
from .block import Block
from mizore import np_array
from jax.numpy import pi

from ...operators import QubitOperator


class Rotation(Block):
    """
    Implement exp(iPt/2)
    """

    def __init__(self, qset, pauli_ops, weight=1, angle_shift=None):
        self.qset = qset
        self.pauli_ops = pauli_ops
        self.weight = weight
        Block.__init__(self, 1, fixed_param=angle_shift)

    def get_gates(self, params):
        return [PauliRotation(self.qset, self.pauli_ops, self.weight * (self._fixed_param + params[0]))]

    def get_inverse_block(self) -> Rotation:
        return Rotation(self.qset, self.pauli_ops, self.weight, self.fixed_param)

    def get_gradient_blocks(self, param_index: int, params=None):
        return [
            (0.5 * self.weight,
             Rotation(self.qset, self.pauli_ops, self.weight, angle_shift=pi / (2 * self.weight) + self.fixed_param)),
            (-0.5 * self.weight,
             Rotation(self.qset, self.pauli_ops, self.weight, angle_shift=-pi / (2 * self.weight) + self.fixed_param))]

    @classmethod
    def get_rotation_blocks(cls, qubit_operator: QubitOperator):
        gates = []
        for qset_op_weight in qubit_operator.qset_op_weight():
            gates.append(*qset_op_weight)
        return gates
