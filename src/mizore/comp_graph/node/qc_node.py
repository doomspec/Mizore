from copy import copy
from typing import Union, List

from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators.observable import Observable

from mizore.comp_graph.comp_param import CompParam
from mizore.comp_graph.comp_node import CompNode


default_random_config = {
    # The number of experiment when running probabilistic qtask
    # Will have no effect when the backend_circuit is not probabilistic
    "use_dm": True,
    # Whether to use density matrix to simulate probabilistic qtask
    "n_exp": 1000
}


class QCircuitNode(CompNode):

    def __init__(self, circuit: MetaCircuit, obs: Union[List[Observable]], name=None, random_config=None):
        super().__init__(name=name)
        self.circuit: MetaCircuit = circuit
        self._obs: List[Observable]
        self.single_obs = False
        if isinstance(obs, list):
            self._obs = obs
        elif isinstance(obs, Observable):
            self.single_obs = True
            self._obs = [obs]
        else:
            raise TypeError()
        self.add_output_param("ExpMean", CompParam(name=f"{self.name}-ExpMean"))
        self.random_config = random_config if random_config is not None else copy(default_random_config)

    @property
    def obs(self):
        return self._obs

    @property
    def exp_mean(self):
        return self.outputs["ExpMean"]

    def __call__(self, *args, **kwargs):
        """
        :return: Two CompParam representing the value of the expectation value and the variance of it
        """
        return self.outputs["ExpMean"]
