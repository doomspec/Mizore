import time
from collections import Counter
from typing import Dict

from mizore.comp_graph.comp_graph import GraphIterator
from mizore.comp_graph.node.dc_node import DeviceCircuitNode
from mizore.operators import QubitOperator
from mizore.transpiler.measurement.abst_measure import MeasureTranspiler, MeasureImpl
from mizore.transpiler.measurement.policy.policy import UniversalPolicy
from mizore.transpiler.transpiler import Transpiler

import numpy as np
import jax.numpy as jnp

UniversalMeasureName = "UniversalMeasure"

EXACT = "exact"
AVERAGE = "average"
DIAGONAL = "diagonal"


class UniversalMeasurePrep(Transpiler):
    def __init__(self, policy_maker, cov_approx, policy_cache):
        super().__init__()
        self.cov_approx = cov_approx
        self.policy_maker = policy_maker
        self.policy_cache: Dict[QubitOperator, UniversalPolicy] = policy_cache

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            self.transpile_node(node)

    def transpile_node(self, node):
        obs = node.obs
        if obs in self.policy_cache:
            policy = self.policy_cache[obs]
        else:
            policy = self.policy_maker(obs)
            self.policy_cache[obs] = policy

        if self.cov_approx == EXACT:
            pass
        elif self.cov_approx == AVERAGE or self.cov_approx == DIAGONAL:
            pass
            if self.cov_approx == "diagonal":
                pass


pauli_op_to_prime = {
    "X": 2,
    "Y": 3,
    "Z": 5
}


def get_prime_pword_tensor(pwords: list, n_qubit):
    pword_tensors = []
    for pword in pwords:
        prime_repr = [0] * n_qubit
        for i_qubit, op in pword:
            prime_repr[i_qubit] = pauli_op_to_prime[op]
        pword_tensors.append(prime_repr)
    return jnp.array(pword_tensors)


def measure_res_for_pwords(pword_in_prime_tensors, pword_measured_in_complement):
    # In prod, only 30, -30, and 0 are valid data. We should eliminate other values.
    prod = pword_in_prime_tensors * pword_measured_in_complement
    is_qwc = 1 - jnp.sign(jnp.sum(jnp.abs(jnp.abs(jnp.abs(prod) - 15) - 15), axis=1))
    minus_30_count = jnp.sum(1 - jnp.abs(jnp.sign(prod + 30)), axis=1)
    measure_res = is_qwc * jnp.power(-1, minus_30_count)
    return measure_res


class UniversalMeasureImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot, policy):
        super().__init__(observable, n_shot)
        self.policy: UniversalPolicy = policy

    def estimate_by_state(self, state):
        # Prepare the accumulators
        n_qubit = self.policy.n_qubit
        pwords_from_each_head = self.policy.sample_pwords(self.n_shot, time.time_ns())
        hamil_pword_cover_count = {}
        hamil_pword_res = {}
        for i_head in range(len(self.policy.heads_children)):
            children = self.policy.heads_children[i_head]
            children_prime_repr = get_prime_pword_tensor(children, n_qubit)
            children_res = jnp.zeros((len(children),))
            children_cover_count = jnp.zeros((len(children),))
            pwords_tuples = [tuple(int(p) for p in pw) for pw in pwords_from_each_head[i_head]]
            if len(pwords_tuples) == 0:
                continue
            pword_counts = Counter(pwords_tuples)
            res_list = []
            for pword_tuple, n_shot in pword_counts.items():
                res = state.sample_pauli_measure_by_pauliword_nums(pword_tuple, n_shot, seed=time.time_ns())
                res_list.append(jnp.array(res))
            measure_res_list = jnp.concatenate(res_list, axis=0)

            for res in measure_res_list:
                res_num_for_pwords = measure_res_for_pwords(children_prime_repr, res)
                children_res += res_num_for_pwords
                children_cover_count += jnp.abs(res_num_for_pwords)

            for i_children in range(len(children)):
                child_tuple = children[i_children]
                if child_tuple in hamil_pword_res:
                    hamil_pword_res[child_tuple] += children_res[i_children]
                    hamil_pword_cover_count[child_tuple] += children_cover_count[i_children]
                else:
                    hamil_pword_res[child_tuple] = children_res[i_children]
                    hamil_pword_cover_count[child_tuple] = children_cover_count[i_children]
        estimation = 0.0
        for pword, res in hamil_pword_res.items():
            cover_count = hamil_pword_cover_count[pword]
            if cover_count != 0:
                estimation += res / cover_count * self.ob.terms[pword]
        return estimation


class UniversalMeasure(MeasureTranspiler):
    def __init__(self, policy_maker, cov_approx="exact"):
        super().__init__()
        self.cov_approx = cov_approx
        self.policy_maker = policy_maker
        self.policy_cache = {}

    def measure_impl(self, observable: QubitOperator, n_shot):
        if observable in self.policy_cache:
            policy = self.policy_cache[observable]
        else:
            policy = self.policy_maker(observable)
        return UniversalMeasureImpl(observable, n_shot, policy)

    def prepare(self):
        return UniversalMeasurePrep(self.policy_maker, self.cov_approx, self.policy_cache)

    def transpile(self, graph_iterator: GraphIterator):
        node: DeviceCircuitNode
        for node in graph_iterator.by_type(DeviceCircuitNode):
            var_coeff = 1.0
            if UniversalMeasureName not in node.aux_info_dict:
                raise Exception()
            cov_approx = self.cov_approx
            if UniversalMeasureName not in node.aux_obs_dict:
                cov_approx = "average"
            aux_info_dict = node.aux_info_dict[UniversalMeasureName]
            multiplier_list = np.array(aux_info_dict["multiplier_list"])
            if cov_approx == "exact":
                op_prod_expv_list = np.array(node.aux_obs_dict["BiasedShadow"]["res"])
                print(sum(multiplier_list), sum(op_prod_expv_list), len(op_prod_expv_list))
                var_coeff = np.sum(multiplier_list * op_prod_expv_list)
                var_coeff -= (node.expv - node.obs.constant) ** 2

            node.expv.set_to_random_variable(var_coeff / node.shot_num, check_valid=False)


if __name__ == '__main__':
    from mizore.transpiler.measurement.policy.L1 import L1_policy_maker
    from chemistry.simple_mols import simple_4_qubit_lih, large_12_qubit_lih
    from mizore.backend_circuit.backend_state import BackendState

    hamil, _ = large_12_qubit_lih().remove_constant()
    n_qubit = 12
    policy = L1_policy_maker(hamil, n_qubit)
    n_shot = 1000
    impl = UniversalMeasureImpl(hamil, n_shot, policy)
    mean, var = impl.get_mean_and_variance_from_multi_exp(3, BackendState(n_qubit))
    print(mean, var * n_shot)
