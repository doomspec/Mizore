from __future__ import annotations

from typing import List, Union


class Gate:
    def __init__(self, qset):
        self.qset = qset
        self.is_noise = False

    @property
    def qulacs_gate(self):
        return

    def simple_reduce(self) -> Union[List[Gate], None]:
        return None

    def get_inverse(self):
        return None

    def __str__(self):
        return "{} at {}".format(str(type(self).__name__), self.qset)
