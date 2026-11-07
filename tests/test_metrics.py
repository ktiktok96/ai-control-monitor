import math

from control_eval.metrics import auroc, bootstrap_ci, catch_rate, threshold_at_fpr


def test_perfect_separation():
    honest = [1, 2, 3, 4, 5]
    attack = [90, 95, 99]
    assert auroc(honest, attack) == 1.0
    assert catch_rate(honest, attack, fpr=0.0) == 1.0


def test_no_separation():
    xs = [10, 20, 30, 40]
    assert auroc(xs, xs) == 0.5


def test_threshold_respects_fpr():
    honest = list(range(100))  # 0..99
    t = threshold_at_fpr(honest, 0.05)
    assert sum(h > t for h in honest) / len(honest) <= 0.05


def test_catch_rate_partial():
    honest = [0] * 19 + [50]  # one honest run looks suspicious
    attack = [40, 60, 80, 100]
    # 5% FPR on 20 runs allows 1 above threshold -> threshold = 0
    assert catch_rate(honest, attack, fpr=0.05) == 1.0
    # 0% FPR -> threshold = 50, only attacks > 50 caught
    assert catch_rate(honest, attack, fpr=0.0) == 0.75


def test_empty_attack_is_nan():
    assert math.isnan(catch_rate([1, 2], []))


def test_bootstrap_ci_brackets_point_estimate():
    honest = [5, 10, 15, 20, 25, 30, 35, 40]
    attack = [20, 45, 50, 60, 70, 15]
    lo, hi = bootstrap_ci(honest, attack, fpr=0.1, n=500)
    assert lo <= catch_rate(honest, attack, fpr=0.1) <= hi
