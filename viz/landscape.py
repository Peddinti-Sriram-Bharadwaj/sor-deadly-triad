"""Optimisation geometry of semi-gradient Q-learning on the counterexamples: SGD vs Adam.

Semi-gradient TD is not gradient descent on a fixed loss: the target moves with the parameters,
so its update field is in general not the gradient of any function. The figures therefore plot
the update field against a *fixed* objective, the value error

    E(theta) = sum_{s,a} d(s,a) Q_theta(s,a)^2      (Q* = 0 in both problems),

on which the field can point uphill.

  fig_plane_<problem>.png  2-D plane through the initial parameters, spanned by the first two
                           principal components of the SGD and Adam trajectories (Li et al.,
                           2018, Sec. 7): log10 E as background, the expected TD update projected
                           onto the plane as arrows, and both trajectories.
  fig_slices_<problem>.png 1-D slices of E and of the frozen-target TD loss along a random,
                           filter-normalised direction (Li et al., 2018), at the initial and the
                           final parameters of each optimiser.
  fig_drift.png            max |Q| and parameter distance from the initialisation against
                           updates, for SGD and Adam on Tsitsiklis & Van Roy, with the predicted
                           growth laws (Adam: ||theta - theta0|| ~ lr t, max |Q| ~ (lr t)^2).

    python viz/landscape.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sor_triad import PROBLEMS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
GAMMA, HIDDEN, DIVERGED = 0.99, 32, 1e6


def make_net(prob, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(prob["phi_s"].shape[1], HIDDEN), torch.nn.ReLU(),
                               torch.nn.Linear(HIDDEN, prob["A"]))


def flat(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def set_flat(net, v):
    i = 0
    for p in net.parameters():
        p.data.copy_(v[i:i + p.numel()].view_as(p))
        i += p.numel()


class Problem:
    """Expected quantities of a problem for a given network (no sampling noise)."""

    def __init__(self, prob, w=1.0):
        self.x = torch.as_tensor(prob["phi_s"], dtype=torch.float32)
        self.P = torch.as_tensor(prob["P"], dtype=torch.float32)
        self.R = torch.as_tensor(prob["R"], dtype=torch.float32)
        self.d = torch.as_tensor(prob["d"], dtype=torch.float32)
        self.w = w

    def value_error(self, net):
        with torch.no_grad():
            return float((self.d * net(self.x) ** 2).sum())

    def td_loss(self, net, frozen=None):
        """0.5 sum d (y - Q)^2 with y computed from `frozen` (default: the network itself)."""
        q = net(self.x)
        with torch.no_grad():
            qf = (frozen or net)(self.x)
            v = qf.max(1).values
            y = self.w * (self.R + GAMMA * torch.einsum("sat,t->sa", self.P, v)) + (1 - self.w) * v[:, None]
        return 0.5 * (self.d * (y - q) ** 2).sum()

    def expected_update(self, net):
        """Expected semi-gradient TD direction (the negative frozen-target gradient)."""
        net.zero_grad()
        self.td_loss(net).backward()
        return -torch.cat([p.grad.reshape(-1) for p in net.parameters()])


def train(prob_name, optimizer, lr, steps, seed=0, every=50, batch=32):
    """The repository's neural learner, recording parameters every `every` updates."""
    prob = PROBLEMS[prob_name]()
    rng = np.random.default_rng(seed)
    net = make_net(prob, seed)
    opt = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer](net.parameters(), lr=lr)
    S, A, P = prob["S"], prob["A"], prob["P"]
    x = torch.as_tensor(prob["phi_s"], dtype=torch.float32)
    p_sa = (prob["d"] / prob["d"].sum()).reshape(-1)
    with torch.no_grad():
        traj, qmax, steps_rec = [flat(net)], [float(net(x).abs().max())], [0]
    for t in range(1, steps + 1):
        idx = rng.choice(S * A, size=batch, p=p_sa)
        s, a = idx // A, idx % A
        s2 = np.minimum((rng.random(batch)[:, None] > P[s, a].cumsum(1)).sum(1), S - 1)
        q = net(x)
        with torch.no_grad():
            v = q.max(1).values
            y = torch.as_tensor(prob["R"][s, a], dtype=torch.float32) + GAMMA * v[s2]
        loss = ((q[torch.as_tensor(s), torch.as_tensor(a)] - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % every == 0:
            with torch.no_grad():
                m = float(net(x).abs().max())
            if not np.isfinite(m) or m > DIVERGED:
                break
            traj.append(flat(net)); qmax.append(m); steps_rec.append(t)
    return {"traj": torch.stack(traj), "qmax": np.array(qmax), "steps": np.array(steps_rec), "net": net}


def filter_normalised_direction(net, seed):
    """Random direction with each filter (row of each weight; biases as a whole) rescaled to the
    norm of the corresponding filter of the network (Li et al., 2018)."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for p in net.parameters():
        r = torch.randn(p.shape, generator=g)
        if p.dim() > 1:
            r = r * (p.detach().norm(dim=1, keepdim=True) / (r.norm(dim=1, keepdim=True) + 1e-10))
        else:
            r = r * (p.detach().norm() / (r.norm() + 1e-10))
        parts.append(r.reshape(-1))
    return torch.cat(parts)


def plane(prob_name, runs, problem, n=41):
    prob = PROBLEMS[prob_name]()
    theta0 = runs[0]["traj"][0]
    M = torch.cat([r["traj"] - theta0 for r in runs])
    _, _, Vt = torch.linalg.svd(M, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    coords = [((r["traj"] - theta0) @ torch.stack([d1, d2]).T).numpy() for r in runs]
    allc = np.concatenate(coords)
    pad = 0.15 * (allc.max(0) - allc.min(0) + 1e-9)
    a_grid = np.linspace(allc[:, 0].min() - pad[0], allc[:, 0].max() + pad[0], n)
    b_grid = np.linspace(allc[:, 1].min() - pad[1], allc[:, 1].max() + pad[1], n)
    net = make_net(prob, 0)
    E = np.zeros((n, n)); U = np.zeros((n, n)); V = np.zeros((n, n))
    for i, b in enumerate(b_grid):
        for j, a in enumerate(a_grid):
            set_flat(net, theta0 + a * d1 + b * d2)
            E[i, j] = problem.value_error(net)
            u = problem.expected_update(net)
            U[i, j], V[i, j] = float(u @ d1), float(u @ d2)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cs = ax.contourf(a_grid, b_grid, np.log10(E + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 value error  Σ d Q²")
    step = max(1, n // 15)
    mag = np.hypot(U, V) + 1e-12
    ax.quiver(a_grid[::step], b_grid[::step], (U / mag)[::step, ::step], (V / mag)[::step, ::step],
              color="white", alpha=0.7, scale=25, width=0.004)
    styles = {"sgd": ("tab:red", "SGD"), "adam": ("tab:orange", "Adam")}
    for r, c in zip(runs, coords):
        color, label = styles[r["optimizer"]]
        ax.plot(c[:, 0], c[:, 1], "-", color=color, lw=2, label=f"{label} (lr {r['lr']:g})")
        ax.plot(c[-1, 0], c[-1, 1], "o", color=color, ms=7, mec="k")
    ax.plot(0, 0, "k*", ms=12, label="initialisation")
    ax.set_xlabel("PC 1 of trajectories"); ax.set_ylabel("PC 2 of trajectories")
    ax.set_title(f"{prob_name}: value error (background) vs expected TD update (arrows)", fontsize=9)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_plane_{prob_name}.png"), dpi=160)
    plt.close(fig)


def slices(prob_name, runs, problem, n=81, seed=1):
    prob = PROBLEMS[prob_name]()
    alphas = np.linspace(-1, 1, n)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    net, probe = make_net(prob, 0), make_net(prob, 0)
    points = [("init", runs[0]["traj"][0], "k")] + [(f"{r['optimizer']} end", r["traj"][-1],
                                                    "tab:red" if r["optimizer"] == "sgd" else "tab:orange") for r in runs]
    for label, theta, color in points:
        set_flat(net, theta)
        dvec = filter_normalised_direction(net, seed)
        E, L = [], []
        for a in alphas:
            set_flat(probe, theta + a * dvec)
            E.append(problem.value_error(probe))
            L.append(float(problem.td_loss(probe, frozen=net)))  # target frozen at the centre
        axes[0].semilogy(alphas, np.array(E) + 1e-12, color=color, label=label)
        axes[1].semilogy(alphas, np.array(L) + 1e-12, color=color, label=label)
    axes[0].set_title("value error along a filter-normalised direction", fontsize=9)
    axes[1].set_title("TD loss with target frozen at the centre point", fontsize=9)
    for ax in axes:
        ax.set_xlabel("step along direction"); ax.legend(fontsize=7)
    fig.suptitle(prob_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_slices_{prob_name}.png"), dpi=160)
    plt.close(fig)


def drift(steps):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    lines = []
    for opt, lr, color in [("sgd", 1e-3, "tab:red"), ("adam", 1e-4, "tab:orange"), ("adam", 1e-3, "tab:brown")]:
        r = train("tvr", opt, lr, steps, every=20)
        keep = r["steps"] > 0  # log axes: start at the first recorded update
        dist = (r["traj"] - r["traj"][0]).norm(dim=1).numpy()[keep]
        t, r["qmax"], r["steps"] = r["steps"][keep], r["qmax"][keep], r["steps"][keep]
        axes[0].loglog(t, r["qmax"], color=color, label=f"{opt} lr {lr:g}")
        axes[1].loglog(t, dist + 1e-12, color=color, label=f"{opt} lr {lr:g}")
        if opt == "adam":
            late = r["steps"] > steps // 4
            slope_q = np.polyfit(np.log(t[late]), np.log(r["qmax"][late]), 1)[0]
            slope_d = np.polyfit(np.log(t[late]), np.log(dist[late]), 1)[0]
            lines.append(f"Adam lr {lr:g}: max|Q| ~ t^{slope_q:.2f}, ||θ-θ0|| ~ t^{slope_d:.2f}, "
                         f"final max|Q| = {r['qmax'][-1]:.3g}")
        else:
            lines.append(f"SGD lr {lr:g}: diverged after {r['steps'][-1]} updates" if r["steps"][-1] < steps
                         else f"SGD lr {lr:g}: final max|Q| = {r['qmax'][-1]:.3g}")
    axes[0].set_title("max |Q| (Q* = 0)", fontsize=9); axes[1].set_title("‖θ − θ0‖", fontsize=9)
    for ax in axes:
        ax.set_xlabel("updates"); ax.legend(fontsize=8)
    fig.suptitle("Tsitsiklis & Van Roy, neural Q: SGD diverges, Adam drifts polynomially", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_drift.png"), dpi=160)
    plt.close(fig)
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    notes = []
    for name, sgd_lr, adam_lr, steps in [("tvr", 1e-3, 1e-4, 20_000), ("baird", 1e-2, 1e-3, 20_000)]:
        prob = PROBLEMS[name]()
        problem = Problem(prob)
        runs = []
        for opt, lr in (("sgd", sgd_lr), ("adam", adam_lr)):
            r = train(name, opt, lr, steps)
            r.update(optimizer=opt, lr=lr)
            runs.append(r)
            notes.append(f"{name} {opt} lr {lr:g}: {len(r['steps']) - 1} recorded points, last update "
                         f"{r['steps'][-1]}, final max|Q| {r['qmax'][-1]:.3g}, final value error "
                         f"{problem.value_error(r['net']):.3g}")
        plane(name, runs, problem)
        slices(name, runs, problem)
    notes += drift(20_000)
    with open(os.path.join(OUT, "notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
