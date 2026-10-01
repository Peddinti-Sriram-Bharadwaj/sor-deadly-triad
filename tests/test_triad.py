import numpy as np
import pytest

from sor_triad import baird, expected_run, tsitsiklis_van_roy
from sor_triad.problems import features


def expected_tvr_drift(gamma, w):
    """d theta / theta for expected linear TD on the Tsitsiklis-Van Roy example: w (6 gamma - 5) / 2."""
    return w * (6 * gamma - 5) / 2


@pytest.mark.parametrize("w", [0.3, 1.0, 2.0])
def test_tvr_matches_closed_form(w):
    # One expected update from theta = 1 changes theta by alpha * w * (6 gamma - 5) / 2.
    r = expected_run(tsitsiklis_van_roy(), "linear", w, alpha=0.01, steps=1, gamma=0.99)
    q2 = 2 * (1 + 0.01 * expected_tvr_drift(0.99, w))  # Q(s2) = 2 theta after one step
    assert np.isclose(r["max_abs_q"], q2)


@pytest.mark.parametrize("gamma,diverges", [(0.8, False), (0.99, True)])
def test_tvr_divergence_threshold_is_independent_of_w(gamma, diverges):
    for w in (0.3, 1.0, 2.0):
        r = expected_run(tsitsiklis_van_roy(), "linear", w, alpha=0.5, steps=20_000, gamma=gamma)
        assert r["diverged"] == diverges


def test_baird_features_and_visits():
    p = baird()
    assert np.isclose(p["d"].sum(), 1.0) and np.allclose(p["d"].sum(1), 1 / 7)
    assert np.allclose(p["P"].sum(-1), 1.0)
    phi, theta = features(p, "linear")
    assert phi.shape == (7, 2, 16) and theta.shape == (16,)


def test_tabular_baird_converges_to_zero():
    r = expected_run(baird(), "tabular", 1.0, alpha=0.5, steps=200_000)
    assert not r["diverged"] and r["max_abs_q"] < 1e-3


def test_linear_baird_diverges_for_q_learning():
    assert expected_run(baird(), "linear", 1.0, alpha=0.1, steps=20_000)["diverged"]
