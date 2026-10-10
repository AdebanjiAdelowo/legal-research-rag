import math

import numpy as np

from legalrag.eval import metrics as M
from legalrag.eval import stats as S

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_precision_hit():
    assert M.recall_at_k(RANKED, {"b", "z"}, 3) == 0.5
    assert M.precision_at_k(RANKED, {"b", "c"}, 5) == 0.4
    assert M.precision_at_k(["a"], {"a"}, 10) == 0.1  # short lists are not rewarded
    assert M.hit_at_k(RANKED, {"e"}, 4) == 0.0 and M.hit_at_k(RANKED, {"e"}, 5) == 1.0


def test_mrr_and_ndcg():
    assert M.reciprocal_rank(RANKED, {"c"}, 10) == 1 / 3
    assert M.reciprocal_rank(RANKED, {"c"}, 2) == 0.0
    assert M.ndcg_at_k(RANKED, {"a"}, 10) == 1.0
    assert math.isclose(M.ndcg_at_k(RANKED, {"b"}, 10), 1 / math.log2(3))
    two = (1 / math.log2(2) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert math.isclose(M.ndcg_at_k(RANKED, {"a", "c"}, 10), two)
    assert M.ndcg_at_k(RANKED, set(), 10) == 0.0


def test_bootstrap_interval_contains_mean_and_is_seeded():
    x = np.random.default_rng(1).random(300)
    mean, low, high = S.bootstrap_ci(x, seed=3)
    assert low < mean < high and high - low < 0.1
    assert S.bootstrap_ci(x, seed=3) == (mean, low, high)


def test_paired_tests_detect_a_real_difference_only():
    rng = np.random.default_rng(0)
    base = rng.random(400)
    assert S.paired_permutation(base + 0.05, base) < 0.001
    noise = base + rng.normal(0, 0.05, 400)
    assert S.paired_permutation(noise, base) > 0.01
    assert S.paired_permutation(base, base) == 1.0
    diff, low, high = S.paired_bootstrap(noise + 0.05, base)
    assert low < diff < high and low < 0.05 < high


def test_holm_is_monotone_and_bounded():
    adj = S.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == [0.04, 0.09, 0.09, 0.5]
    assert S.holm([0.9, 0.9]) == [1.0, 1.0]


def test_wilson_and_auroc():
    p, low, high = S.wilson(8, 10)
    assert p == 0.8 and 0.49 < low < 0.5 and 0.94 < high < 0.95
    assert S.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert S.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert S.auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
