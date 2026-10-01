"""Problem definitions. Each problem is a dict with

    S, A     number of states and actions
    phi_s    state features [S, k] used by the linear and neural learners
    P        transition probabilities [S, A, S]
    R        expected rewards [S, A]
    d        state-action weighting of the updates (the behaviour distribution) [S, A]
    theta0   initial linear weights per action block [k]

All rewards are zero, so Q* = 0 and every relaxed fixed point w Q* + (1 - w) V* is also 0.
"""

import numpy as np


def tsitsiklis_van_roy():
    """Two states with features 1 and 2; s1 -> s2 -> s2, uniform weighting, one action.

    Expected linear TD(0) diverges for gamma > 5/6 (Tsitsiklis & Van Roy, 1997; Sutton & Barto,
    2018, Example 11.1 with no termination). With one action, max_c Q(s, c) = Q(s), so the SOR
    target reduces to a rescaled TD error.
    """
    P = np.zeros((2, 1, 2))
    P[0, 0, 1] = P[1, 0, 1] = 1.0
    return {"S": 2, "A": 1, "phi_s": np.array([[1.0], [2.0]]), "P": P, "R": np.zeros((2, 1)),
            "d": np.full((2, 1), 0.5), "theta0": np.array([1.0])}


def baird():
    """Baird's star (Baird, 1995) in a two-action control form.

    Six upper states and one lower state with the features of Sutton & Barto (2018, Fig. 11.1):
    upper state i has 2 at feature i and 1 at feature 8; the lower state has 1 at feature 7 and
    2 at feature 8. "dashed" moves to a uniformly random upper state, "solid" to the lower state,
    which therefore loops on itself under solid. The behaviour policy takes dashed with
    probability 6/7, so every state is visited with probability 1/7. Q is linear in a separate
    copy of the features per action; both copies start at (1, 1, 1, 1, 1, 1, 10, 1).
    """
    S, A = 7, 2
    phi_s = np.zeros((S, 8))
    for s in range(6):
        phi_s[s, s], phi_s[s, 7] = 2.0, 1.0
    phi_s[6, 6], phi_s[6, 7] = 1.0, 2.0
    P = np.zeros((S, A, S))
    P[:, 0, :6] = 1 / 6
    P[:, 1, 6] = 1.0
    d = np.full((S, A), 1 / 7) * np.array([6 / 7, 1 / 7])
    return {"S": S, "A": A, "phi_s": phi_s, "P": P, "R": np.zeros((S, A)), "d": d,
            "theta0": np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)}


PROBLEMS = {"tvr": tsitsiklis_van_roy, "baird": baird}


def features(prob, representation):
    """Returns phi [S, A, n] and initial weights [n] for 'tabular' or 'linear'.

    Tabular uses one weight per (s, a), initialised to the same Q-values as the linear learner.
    """
    S, A, phi_s, theta0 = prob["S"], prob["A"], prob["phi_s"], prob["theta0"]
    if representation == "tabular":
        q0 = phi_s @ theta0
        return np.eye(S * A).reshape(S, A, S * A), np.repeat(q0[:, None], A, 1).reshape(-1)
    if representation == "linear":
        k = phi_s.shape[1]
        phi = np.zeros((S, A, A * k))
        for a in range(A):
            phi[:, a, a * k:(a + 1) * k] = phi_s
        return phi, np.tile(theta0, A)
    raise ValueError(f"unknown representation {representation!r}")
