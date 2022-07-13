from mizore.testing.random_operator import get_random_operator


def test():
    weight_sum_expected = 1.0
    n_term_expected = 10
    for seed in range(1, 100):
        hamil = get_random_operator(4, n_term_expected, weight_sum_expected, seed)
        weight_sum = 0.0
        n_term = 0
        for qset, op, weight in hamil.qset_op_weight():
            n_term += 1
            weight_sum += abs(weight)
        assert abs(weight_sum - weight_sum_expected) < 1e-8
        assert n_term == n_term_expected
