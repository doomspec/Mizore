from copy import copy
from typing import Union, List, Iterable

from mizore.backend_circuit.backend_state import BackendState
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator

from mizore.comp_graph.value import Value
from mizore.comp_graph.comp_node import CompNode
from mizore import jax_array

default_config = {
    # The number of experiment when running probabilistic qtask
    # Will have nO effect when the backend_circuit is not probabilistic
    "use_dm": True,
    # Whether to use density matrix to simulate probabilistic qtask
    "n_exp": 1000
}


class QCircuitNode(CompNode):

    def __init__(self, circuit: MetaCircuit, obs: QubitOperator, name=None, config=None):
        super().__init__(name=name)
        self._circuit: MetaCircuit = circuit
        self._obs: QubitOperator
        self._obs = obs

        self.add_output_value(f"ExpValue")

        self.add_input_value("Params", jax_array([0.0] * circuit.n_param))
        self.config = config if config is not None else copy(default_config)

        self.aux_obs_dict = {}  # {"key":{"obs":[],"res":[],"config":{}}}
        self.aux_info_dict = {}
        # self.save_state = False
        # self.saved_state: Union[BackendState, None] = None

    @property
    def circuit(self) -> MetaCircuit:
        return self._circuit

    @property
    def obs(self):
        return self._obs

    @property
    def params(self) -> Value:
        return self.inputs["Params"]

    @property
    def expv(self):
        return self.outputs["ExpValue"]

    def __call__(self, *args, **kwargs):
        return self.expv
