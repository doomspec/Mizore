import pytest

from mizore.backend_circuit.rotations import SingleRotation, PauliRotation
from math import pi
from qulacs.gate import to_matrix_gate
from numpy.testing import assert_array_almost_equal
from math import cos, sin


@pytest.fixture
def angles():
    return [1.0, 1.5, 2.0, 3.0, 4.0]


def test_single_rotation(angles):
    for angle in angles:
        matrix = to_matrix_gate(SingleRotation(1, 0, angle).qulacs_gate).get_matrix()
        matrix_expect = [[cos(angle / 2), 1j * sin(angle / 2)], [1j * sin(angle / 2), cos(angle / 2)]]
        assert_array_almost_equal(matrix, matrix_expect)


def test_multi_rotation(angles):
    for angle in angles:
        matrix = to_matrix_gate(SingleRotation(1, 0, angle).qulacs_gate).get_matrix()
        matrix_expect = to_matrix_gate(PauliRotation([0], [1], angle).qulacs_gate).get_matrix()
        assert_array_almost_equal(matrix, matrix_expect)
