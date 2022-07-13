from typing import Dict

from mizore.backend_circuit.backend_state import BackendState


class StateProcessor:
    def __init__(self, key):
        self.key = key

    def load_node(self, node):
        pass

    def process(self, state: BackendState) -> any:
        pass

    def post_process(self, node, process_res):
        pass