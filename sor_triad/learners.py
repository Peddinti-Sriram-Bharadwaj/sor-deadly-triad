"""Semi-gradient Q-learning with the SOR target

    y_w(s, a) = w * (r + gamma * max_a' Q(s', a')) + (1 - w) * max_c Q(s, c),

w = 1 being Q-learning (Kamanchi et al., 2020). The target is held fixed when differentiating
(no target network), so the setting has all three elements of the deadly triad: function
approximation, bootstrapping and off-policy updates.
"""

import numpy as np

from .problems import features

DIVERGED = 1e6


def sor_target(q, r, gamma, next_v, w):
    """y_w for arrays of current Q-values q [..., A] (per state), rewards and next-state values."""
    return w * (r + gamma * next_v) + (1 - w) * q.max(-1, keepdims=True)


def expected_run(prob, representation, w, alpha, steps, gamma=0.99):
    """Expected (synchronous) updates: theta += alpha * sum_{s,a} d(s,a) phi(s,a) (E[y_w] - Q(s,a)).

    Returns max |Q| at the end (the error, since Q* = 0), whether it diverged, and when.
    """
    phi, theta = features(prob, representation)
    P, R, d = prob["P"], prob["R"], prob["d"]
    for t in range(steps):
        q = phi @ theta
        m = float(np.abs(q).max())
        if not np.isfinite(m) or m > DIVERGED:
            return {"max_abs_q": float("inf"), "diverged": True, "step": t}
        y = sor_target(q, R, gamma, P @ q.max(1), w)
        theta = theta + alpha * np.einsum("sak,sa->k", phi, d * (y - q))
    return {"max_abs_q": float(np.abs(phi @ theta).max()), "diverged": False, "step": steps}


def neural_run(prob, w, optimizer, lr, steps, seed, gamma=0.99, batch=32, hidden=32):
    """A one-hidden-layer ReLU network on the state features, trained on sampled minibatches
    drawn from d with sampled next states. optimizer is 'sgd' or 'adam'."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    S, A, P = prob["S"], prob["A"], prob["P"]
    x = torch.as_tensor(prob["phi_s"], dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, A))
    opt = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer](net.parameters(), lr=lr)
    p_sa = (prob["d"] / prob["d"].sum()).reshape(-1)
    for t in range(steps):
        idx = rng.choice(S * A, size=batch, p=p_sa)
        s, a = idx // A, idx % A
        s2 = np.minimum((rng.random(batch)[:, None] > P[s, a].cumsum(1)).sum(1), S - 1)
        q = net(x)
        with torch.no_grad():
            v = q.max(1).values
            y = w * (torch.as_tensor(prob["R"][s, a], dtype=torch.float32) + gamma * v[s2]) + (1 - w) * v[s]
        loss = ((q[torch.as_tensor(s), torch.as_tensor(a)] - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 100 == 0:
            with torch.no_grad():
                m = float(net(x).abs().max())
            if not np.isfinite(m) or m > DIVERGED:
                return {"max_abs_q": float("inf"), "diverged": True, "step": t}
    with torch.no_grad():
        return {"max_abs_q": float(net(x).abs().max()), "diverged": False, "step": steps}
