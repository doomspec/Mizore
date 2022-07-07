from typing import List, Union

from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators.observable import Observable

from mizore.comp_graph.value import Value
from .qc_node import QCircuitNode
from mizore import jax_array, to_jax_array
from ..immutable import Immutable

from ..parameter import Parameter


class DeviceCircuitNode(QCircuitNode):

    def __init__(self, circuit: MetaCircuit, obs: Union[List[Observable], Observable], name=None, config=None,
                 expv_shift_from_var=True, param: Union[Value, None] = None, init_shot_num=10000):
        super().__init__(circuit, obs, name=name, config=config)

        self.add_input_value("ShotNum", Parameter(to_jax_array(init_shot_num*len(self.obs_list))))
        self.add_input_value("Params")

        if param is None:
            self.params.set_value(jax_array([0.0] * circuit.n_param))
        else:
            self.params.bind_to(param)

        self.expv_shift_from_var = expv_shift_from_var

    @property
    def shot_num(self):
        return self.inputs["ShotNum"]

    def __call__(self, *args, **kwargs) -> Value:
        return self.expv
