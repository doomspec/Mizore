from typing import List, Union

from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators.observable import Observable

from mizore.comp_graph.comp_param import CompParam
from .qc_node import QCircuitNode
from mizore.comp_graph.valvar import ValVar
from mizore import jax_array

from ..var_param import VariableParam


class MetaCircuitNode(QCircuitNode):

    def __init__(self, circuit: MetaCircuit, obs: Union[List[Observable], Observable], name=None, random_config=None,
                 calc_shift_by_param_var=True):
        super().__init__(circuit, obs, name=name, random_config=random_config)

        self.add_input_param("ShotNum", VariableParam(jax_array([5000] * len(self._obs))))

        self.shot_num.name = f"{self.name}-ShotNum"

        self.add_output_param("ExpVar", CompParam(name=f"{self.name}-ExpVar"))

        self.add_input_param("Params", ValVar(None))

        self.calc_shift_by_param_var = calc_shift_by_param_var

    @property
    def params(self) -> ValVar:
        return self.inputs["Params"]

    @property
    def shot_num(self):
        return self.inputs["ShotNum"]

    @property
    def exp_var(self):
        return self.outputs["ExpVar"]

    @property
    def exp_valvar(self):
        return ValVar(self.exp_mean, self.exp_var)

    def __call__(self, *args, **kwargs) -> ValVar:
        """
        :return: Two CompParam representing the value of the expectation value and the variance of it
        """
        if self.single_obs:
            return ValVar.get_by_index(ValVar(self.exp_mean, self.exp_var), 0)
        else:
            return ValVar(self.exp_mean, self.exp_var)
