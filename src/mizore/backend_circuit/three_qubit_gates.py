from typing import Tuple, List

from mizore.backend_circuit.gate import Gate
from qulacs.gate import TOFFOLI


class Toffoli(Gate):
    def __init__(self, control_qset: Tuple, target_index):
        assert len(control_qset) == 2
        assert target_index not in control_qset
        self.control_qset = control_qset
        self.target_index = target_index
        super().__init__((*control_qset, target_index))

    @property
    def qulacs_gate(self):
        return TOFFOLI(self.control_qset[0], self.control_qset[1], self.target_index)

    def simple_reduce(self) -> List[Gate]:
        # TODO
        pass
