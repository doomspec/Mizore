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

    def __init__(self, circuit: MetaCircuit, obs: Union[List[QubitOperator], QubitOperator], name=None, config=None):
        super().__init__(name=name)
        self._circuit: MetaCircuit = circuit
        self._obs: List[QubitOperator]
        self._single_obs = False

        self.is_single_obs = False
        self._obs = obs
        if isinstance(obs, List):
            self._obs_list = obs
        else:
            self.is_single_obs = True
            self._obs_list = [obs]

        for i in range(len(self._obs_list)):
            self.add_output_value(f"ExpValue-{i}")

        self.add_input_value("Params", jax_array([0.0] * circuit.n_param))
        self.config = config if config is not None else copy(default_config)

        self.aux_obs_dict = {}  # {"key":{"obs":[],"res":[],"config":{}}}
        #self.save_state = False
        #self.saved_state: Union[BackendState, None] = None

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
        if self.is_single_obs:
            return self.outputs["ExpValue-0"]
        else:
            return self.expv_list()

    def expv_vector(self):
        return Value.array(self.expv_list())

    def expv_list(self) -> List[Value]:
        expv_lst = []
        for i in range(len(self._obs_list)):
            expv_lst.append(self.outputs[f"ExpValue-{i}"])
        return expv_lst

    def iter_expv(self) -> Iterable[Value]:
        for i in range(len(self._obs_list)):
            yield self.outputs[f"ExpValue-{i}"]

    def __call__(self, *args, **kwargs):
        return self.expv


def set_node_expv_list(node, expv_list):
    for i in range(len(node._obs_list)):
        node.outputs[f"ExpValue-{i}"].set_value(expv_list[i])