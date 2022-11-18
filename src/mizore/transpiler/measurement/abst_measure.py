import time
from typing import Optional, Type, Dict, Tuple, List, Callable

from mizore.backend_circuit.backend_state import BackendState
from mizore.operators import QubitOperator
from mizore.transpiler.transpiler import Transpiler
import numpy as np

from tqdm import trange


class MeasureImpl:
    def __init__(self, observable: QubitOperator, n_shot):
        self.ob, self.constant = observable.remove_constant()
        self.n_shot = n_shot

    def get_pauliwords(self) -> List[Tuple]:
        pass

    def get_pauliwords_dense(self) -> Dict[Tuple, int]:
        assert False
        pauliwords = self.get_pauliwords()
        occur_dict = {pauliword: 0 for pauliword in pauliwords}
        for pauliword in pauliwords:
            occur_dict[pauliword] += 1
        return occur_dict

    def one_shot_estimation(self, pauliword, result_qset):
        pass

    def get_result(self, one_shot_estimations):
        assert False
        return sum(one_shot_estimations) / len(one_shot_estimations) + self.constant

    def estimate_by_state(self, state: BackendState):
        assert False
        occur_dict = self.get_pauliwords_dense()
        one_shot_estimations = []
        i = 0
        for pauliword, n_occur in occur_dict.items():
            sampled_qsets = state.sample_pauli_measure_by_pauliword_tuple(pauliword, n_occur,
                                                                          seed=int(time.time()) + i * 11)
            i += 1
            res = [self.one_shot_estimation(pauliword, qset) for qset in sampled_qsets]
            one_shot_estimations.extend(res)
        return np.average(one_shot_estimations)

    def get_mean_and_variance_from_multi_exp(self, n_exp, state: BackendState, pbar=False):
        observed_means = []
        range_ = trange(n_exp) if pbar else range(n_exp)
        for i in range_:
            observed_mean = self.estimate_by_state(state)
            observed_means.append(observed_mean)
        var_experiment = np.var(observed_means, ddof=1)
        mean_experiment = np.mean(observed_means)
        return mean_experiment, var_experiment


class MeasureTranspiler(Transpiler):
    def __init__(self):
        Transpiler.__init__(self, name=self.__class__.__name__)

    def prepare(self):
        pass

    def measure_impl(self, observable: QubitOperator, n_shot):
        pass
