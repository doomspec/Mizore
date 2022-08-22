from typing import List, Union

from mizore.meta_circuit.meta_circuit import MetaCircuit

from mizore.comp_graph.value import Value
from .qc_node import QCircuitNode
from mizore import jax_array, to_jax_array

from ..parameter import Parameter
from ...operators import QubitOperator


class DeviceCircuitNode(QCircuitNode):

    def __init__(self, circuit: MetaCircuit, obs: Union[List[QubitOperator], QubitOperator], name=None, config=None,
                 expv_shift_from_var=True, param: Union[Value, None] = None):
        super().__init__(circuit, obs, name=name, config=config)

        self.add_input_value("ShotNumTotal", only_bind=True)
        self.add_input_value("Params")

        if param is None:
            self.params.set_value(jax_array([0.0] * circuit.n_param))
        else:
            self.params.bind_to(param)

        self.expv_shift_from_var = expv_shift_from_var

        self.shot_num_overwritten = None

    @property
    def shot_num_total(self):
        return self.inputs["ShotNumTotal"]

    @property
    def shot_num(self):
        if self.shot_num_overwritten is None:
            return self.inputs["ShotNumTotal"]
        else:
            return self.shot_num_overwritten(self)

    def __call__(self, *args, **kwargs) -> Value:
        return self.expv
