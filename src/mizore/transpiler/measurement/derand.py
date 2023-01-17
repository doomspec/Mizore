import time
from collections import Counter

import numpy as np

from mizore.operators import QubitOperator
from mizore.transpiler.measurement.abst_measure import MeasureImpl
from mizore.transpiler.measurement.derand_impl import DerandomizationMeasurementNumPy
from mizore.transpiler.measurement.shadowgrouping_impl import ShadowGroupingMeasurement
from mizore.transpiler.measurement.utlis_estimate import estimate_by_list_of_pwords_on_state, map_pwords_to_measure, \
    average_var_by_list_of_pwords, estimate_by_list_of_pwords_on_totally_random_state
from mizore.transpiler.measurement.vectorize_helper import get_prime_pword_tensor, measure_res_for_pwords


class DerandMeasureImpl(MeasureImpl):
    def __init__(self, observable: QubitOperator, n_shot_on_each, max_nshot=None, method="shadowgrouping"):
        super().__init__(observable, n_shot_on_each)
        self.n_qubit = observable.get_qset()[-1] + 1
        if method == "original":
            self.builder = DerandomizationMeasurementNumPy(observable, self.n_qubit)
        else:
            self.builder = ShadowGroupingMeasurement(observable, self.n_qubit)
        pwords_to_measure = self.builder.build(self.n_shot, max_nshot=max_nshot)
        self.pwords_to_measure = pwords_to_measure
        # self.pwords_to_measure = map_pwords_to_measure(pwords_to_measure)

    def estimate_by_state(self, state):
        if state is not None:
            return estimate_by_list_of_pwords_on_state(self.ob, self.pwords_to_measure, state)
        else:
            return estimate_by_list_of_pwords_on_totally_random_state(self.ob, self.pwords_to_measure)


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil
    from mizore.operators.spectrum import get_ground_state

    n_shot = 630  # 2000: 461， 4000: 925
    hamil = get_test_hamil("mol", "LiH_12_BK")
    # hamil, _ = hamil.remove_constant()
    n_qubit = hamil.n_qubit
    time1 = time.time()
    impl = DerandMeasureImpl(hamil, n_shot)
    print(f"Derand generate in {time.time() - time1} seconds")

    energy, state = get_ground_state(n_qubit, hamil)
    mean, var = impl.get_mean_and_variance_from_multi_exp(1000, state, pbar=True)

    # print(energy)
    print(mean, var * n_shot)
    # print(np.sqrt(var * n_shot / 1000))

    average_var = average_var_by_list_of_pwords(hamil, impl.pwords_to_measure)
    print(average_var * n_shot)
