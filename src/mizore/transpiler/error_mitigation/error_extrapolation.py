from copy import deepcopy
from typing import List

from mizore.comp_graph.value import Value
from mizore.comp_graph.value import Variable
from mizore.meta_circuit.block import Block
from mizore.meta_circuit.post_processor import SimpleProcessor
from mizore.backend_circuit.gate import Gate
from mizore.comp_graph.comp_graph import GraphIterator

from mizore.backend_circuit.noise import NoiseGate
from mizore.comp_graph.node.dc_node import DeviceCircuitNode

from mizore.comp_graph.node.qc_node import QCircuitNode

from mizore.transpiler.transpiler import Transpiler

EPSILON = 1e-6


class ErrorExtrapolation(Transpiler):
    """
    Useful reference: https://journals.jps.jp/doi/full/10.7566/JPSJ.90.032001
    """

    def __init__(self, amp_list: List):
        super().__init__()
        self.method = "Richardson"
        # After normalization the amp_list starts from 1.0 and is ascending ordered
        self.amp_list = ErrorExtrapolation.normalize_amp_list(amp_list)

    def transpile(self, graph_iterator: GraphIterator):
        coeffs: List
        if self.method == "Richardson":
            coeffs = ErrorExtrapolation.get_Richardson_coeffs(self.amp_list)
        else:
            raise NotImplementedError()
        node_changed = False
        qcnode: QCircuitNode
        for qcnode in graph_iterator.by_type(QCircuitNode):
            # We skip noiseless qcnode
            if "noisy" not in qcnode.tags:
                print("Not every backend_circuit is noisy")
                continue
            mitigated_value = Value(0.0)
            coeff__square_sum = sum([abs(coeff)**2 for coeff in coeffs])
            for i_level in range(len(self.amp_list)):
                noisy_circuit = qcnode.circuit.replica()
                if self.amp_list[i_level] != 1.0:
                    noisy_circuit.add_post_process(NoiseAmplifier(self.amp_list[i_level]))
                # Construct a new QCircuitNode
                # Note that the shot number of old node will not be inherited
                if isinstance(qcnode, DeviceCircuitNode):
                    noise_qcnode = DeviceCircuitNode(noisy_circuit, qcnode.obs,
                                                     name=qcnode.name + f"-ErrExtrp-{i_level}")
                    noise_qcnode.params.bind_to(qcnode.params)
                else:
                    noise_qcnode = QCircuitNode(noisy_circuit, qcnode.obs,
                                            name=qcnode.name + f"-ErrExtrp-{i_level}")
                noise_qcnode.config = qcnode.config
                result_expv = noise_qcnode()
                # TODO check this: Is this the optimal way to distribute the shot nums?
                noise_qcnode.shot_num.bind_to(qcnode.shot_num*(abs(coeffs[i_level])**2/coeff__square_sum))
                mitigated_value = mitigated_value + (coeffs[i_level] * result_expv)
            mitigated_value.name = "ErrorExtrapExpv"
            qcnode.expv.bind_to(mitigated_value)
            qcnode.expv.set_to_not_random()
            qcnode.in_graph = False
            node_changed = True
        if node_changed:
            graph_iterator.reconstruct_graph()

    @classmethod
    def get_Richardson_coeffs(cls, amp_list):
        """
        See page 18 of https://journals.jps.jp/doi/full/10.7566/JPSJ.90.032001
        \beta_k = \prod_{i\neq k} \frac{a_i}{a_k-a_i}
        :param amp_list:
        :return:
        """
        beta = [0.0] * len(amp_list)
        for k in range(len(amp_list)):
            beta_k = 1 # TODO check here
            for i in range(len(amp_list)):
                if i != k:
                    beta_k *= amp_list[i] / (amp_list[k] - amp_list[i])
            beta[k] = beta_k
        # This is a magic
        # I don't know why this work
        if beta[0] < 0:
            beta = [-x for x in beta]
            pass
        return beta

    @classmethod
    def normalize_amp_list(cls, amp_list) -> List:
        if len(amp_list) < 1:
            raise Exception("Amp list must contain at least one amp. I hope you know what you are doing.")
        amp_list.sort()
        if amp_list[0] <= 1:
            raise Exception("Amp must be larger than 1. I hope you know what you are doing.")
        for i in range(1, len(amp_list)):
            if amp_list[i] - amp_list[i - 1] < EPSILON:
                raise Exception("Amp must be different values. I hope you know what you are doing.")
        amp_list.insert(0, 1.0)
        return amp_list


class NoiseAmplifier(SimpleProcessor):
    def __init__(self, amp_rate):
        self.amp_rate = amp_rate

    def process(self, gates: List[Gate], block: Block):
        noisy_gates = []
        for gate in gates:
            # We amplify each noise
            if isinstance(gate, NoiseGate):
                copied_gate = deepcopy(gate)
                copied_gate.prob = 1 - ((1 - gate.prob) ** self.amp_rate)
                noisy_gates.append(copied_gate)
            else:
                noisy_gates.append(gate)

        return noisy_gates
