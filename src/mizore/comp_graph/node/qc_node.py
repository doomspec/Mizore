from copy import copy
from typing import Union, List

from mizore.comp_graph.immutable import Immutable
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators.observable import Observable

from mizore.comp_graph.value import Value
from mizore.comp_graph.comp_node import CompNode
from mizore import jax_array

default_config = {
    # The number of experiment when running probabilistic qtask
    # Will have no effect when the backend_circuit is not probabilistic
    "use_dm": True,
    # Whether to use density matrix to simulate probabilistic qtask
    "n_exp": 1000
}


class QCircuitNode(CompNode):

    def __init__(self, circuit: MetaCircuit, obs: Union[List[Observable], Observable], name=None, config=None):
        super().__init__(name=name)
        self._circuit: MetaCircuit = circuit
        self._obs: List[Observable]
        self._single_obs = False

        self.is_single_obs = False
        self._obs = obs
        if isinstance(obs, List):
            self._obs_list = obs
        else:
            self.is_single_obs = True
            self._obs_list = [obs]

        self.add_output_value("ExpValue")
        self.add_input_value("Params", jax_array([0.0] * circuit.n_param))
        self.config = config if config is not None else copy(default_config)

    @property
    def circuit(self) -> MetaCircuit:
        return self._circuit

    @property
    def obs(self):
        return self._obs

    @property
    def obs_list(self):
        return self._obs_list

    @property
    def params(self) -> Value:
        return self.inputs["Params"]

    @property
    def expv(self):
        return self.outputs["ExpValue"]

    def __call__(self, *args, **kwargs):
        return self.expv