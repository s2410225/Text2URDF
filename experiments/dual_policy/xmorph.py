"""Cross-morphology dual-policy learning: the better body teaches the other.

Two quadrupeds with different symmetry (radial vs bilateral, see morph.py) each
own a Gaussian PPO policy. They are trained at the same time on the same task
(Ant forward locomotion). Because the bodies differ, their policies can never
become identical, so the difference between them stays informative.

Modes
-----
indep    each body learns alone (baseline)
mutual   both distil toward each other every update with a fixed weight
         (symmetric, ignores who is better: deep-mutual-learning style)
better   the body with the higher recent return is the teacher; the other
         body distils toward the teacher's actions on its *own* states, with a
         weight that grows with the performance gap. Roles can swap at any time.
better_q like `better`, but the student only imitates the teacher on its own
         transitions whose outcome was worse than its critic predicted
         (advantage < 0, i.e. negative prediction error). Where its own action
         beat its expectation it keeps its own, body-specific behaviour.

Distillation target: the teacher's mean action on the student's observation,
normalised with the teacher's own observation statistics. Joint i of one body
is mapped to joint i of the other (same kinematic tree).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

import kp_env
import morph
from dpl import MLP, gae

BODIES = ("radial", "bilateral")
LOG2PI = np.log(2 * np.pi)


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), len(x)
        d, tot = bm - self.mean, self.count + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.count + bv * bn + d**2 * self.count * bn / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


class GaussAgent:
    def __init__(self, obs_dim, act_dim, rng, lr):
        self.mu = MLP([obs_dim, 64, 64, act_dim], rng, out_scale=0.01, lr=lr)
        self.v = MLP([obs_dim, 64, 64, 1], rng, out_scale=1.0, lr=lr)
        self.log_std = np.zeros(act_dim)
        self.ls_m, self.ls_v, self.ls_t = np.zeros(act_dim), np.zeros(act_dim), 0
        self.norm = RunningNorm(obs_dim)
        self.lr = lr

    def mean(self, raw_obs):
        return self.mu.forward(self.norm(raw_obs))

    def step_log_std(self, g, lr):
        self.ls_t += 1
        self.ls_m = 0.9 * self.ls_m + 0.1 * g
        self.ls_v = 0.999 * self.ls_v + 0.001 * g * g
        mh, vh = self.ls_m / (1 - 0.9**self.ls_t), self.ls_v / (1 - 0.999**self.ls_t)
        self.log_std -= lr * mh / (np.sqrt(vh) + 1e-5)


def logp_gauss(a, mu, log_std):
    z = (a - mu) / np.exp(log_std)
    return -0.5 * (z**2).sum(-1) - log_std.sum() - 0.5 * len(log_std) * LOG2PI


def factory(space):
    return kp_env.make if space == "keypoint" else morph.make


class Body:
    def __init__(self, name, n_envs, seed, make):
        self.name = name
        self.envs = [make(name) for _ in range(n_envs)]
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.ep_ret, self.finished = np.zeros(n_envs), []
        self.disc, self.rn = np.zeros(n_envs), RunningNorm(1)

    def step(self, act, gamma):
        rews, dones, nxt = np.zeros(len(self.envs)), np.zeros(len(self.envs)), []
        for i, (e, a) in enumerate(zip(self.envs, act)):
            o, r, te, tr, _ = e.step(np.clip(a, -1, 1))
            self.ep_ret[i] += r
            rews[i] = r
            if te or tr:
                self.finished.append(self.ep_ret[i])
                self.ep_ret[i] = 0
                o, _ = e.reset()
                dones[i] = 1
            nxt.append(o)
        self.obs = np.stack(nxt)
        # reward scaling by running std of discounted return
        self.disc = self.disc * gamma + rews
        self.rn.update(self.disc[:, None])
        self.disc[dones > 0] = 0
        return np.clip(rews / np.sqrt(self.rn.var[0] + 1e-8), -10, 10), dones


def evaluate(agent, body_name, episodes, seed, make):
    env = make(body_name)
    rets, dist = [], []
    for k in range(episodes):
        o, _ = env.reset(seed=50_000 + 97 * seed + k)
        x0 = env.unwrapped.data.qpos[0]
        R, done = 0.0, False
        while not done:
            o, r, te, tr, _ = env.step(np.clip(agent.mean(o[None])[0], -1, 1))
            R += r
            done = te or tr
        rets.append(R)
        dist.append(env.unwrapped.data.qpos[0] - x0)
    return float(np.mean(rets)), float(np.mean(dist))


def train(cfg):
    rng = np.random.default_rng(cfg.seed)
    make = factory(cfg.space)
    bodies = [Body(b, cfg.num_envs, cfg.seed * 1000 + 100 * k, make) for k, b in enumerate(BODIES)]
    obs_dim = bodies[0].obs.shape[1]
    act_dim = bodies[0].envs[0].action_space.shape[0]
    assert all(b.obs.shape[1] == obs_dim and b.envs[0].action_space.shape[0] == act_dim for b in bodies)
    agents = [GaussAgent(obs_dim, act_dim, rng, cfg.lr) for _ in BODIES]
    T, N = cfg.num_steps, cfg.num_envs
    n_updates = cfg.total_steps // (T * N)
    ema = [None, None]
    log = {k: [] for k in ["steps", "ret_radial", "ret_bilateral", "teacher", "w_radial", "w_bilateral",
                           "dist", "qpass", "eval_steps", "eval_radial", "eval_bilateral", "dx_radial", "dx_bilateral"]}
    t0 = time.time()

    for upd in range(n_updates):
        frac = 1 - upd / n_updates
        lr = cfg.lr * frac
        # ---------------- rollouts (raw obs kept so the other policy can read them)
        ros = []
        for ag, bd in zip(agents, bodies):
            O = np.zeros((T, N, obs_dim)); A = np.zeros((T, N, act_dim))
            LP, R, D, V = (np.zeros((T, N)) for _ in range(4))
            for t in range(T):
                o = bd.obs
                ag.norm.update(o)
                mu = ag.mean(o)
                a = mu + np.exp(ag.log_std) * rng.standard_normal(mu.shape)
                O[t], A[t], LP[t] = o, a, logp_gauss(a, mu, ag.log_std)
                V[t] = ag.v.forward(ag.norm(o))[:, 0]
                R[t], D[t] = bd.step(a, cfg.gamma)
            last_v = ag.v.forward(ag.norm(bd.obs))[:, 0]
            adv, ret = gae(R, V, D, last_v, cfg.gamma, cfg.lam)
            ros.append(dict(O=O.reshape(-1, obs_dim), A=A.reshape(-1, act_dim), LP=LP.reshape(-1),
                            adv=adv.reshape(-1), ret=ret.reshape(-1)))

        # ---------------- who is better? (EMA of recent episode returns)
        for k, bd in enumerate(bodies):
            if bd.finished:
                r = float(np.mean(bd.finished[-10:]))
                ema[k] = r if ema[k] is None else 0.9 * ema[k] + 0.1 * r
        w = [0.0, 0.0]
        teacher = -1
        if cfg.mode == "mutual":
            w = [cfg.lam_kl, cfg.lam_kl]
        elif cfg.mode.startswith("better") and None not in ema and upd >= cfg.warmup:
            teacher = int(ema[1] > ema[0])
            s = 1 - teacher
            gap = (ema[teacher] - ema[s]) / (abs(ema[teacher]) + abs(ema[s]) + 1e-8)
            w[s] = cfg.lam_kl * float(np.clip(cfg.gap_gain * gap, 0, 1))

        # teacher targets on the student's states (frozen for this update)
        targets = [None, None]
        for k in range(2):
            if w[k] > 0:
                targets[k] = np.clip(agents[1 - k].mean(ros[k]["O"]), -1, 1)

        # ---------------- PPO + distillation
        dist_log, qpass_log = [np.nan, np.nan], [np.nan, np.nan]
        for k, (ag, ro) in enumerate(zip(agents, ros)):
            B = len(ro["LP"])
            mbs = B // cfg.minibatches
            normO = ag.norm(ro["O"])
            if targets[k] is not None:
                dist_log[k] = float(((ag.mu.forward(normO) - targets[k]) ** 2).sum(-1).mean())
                gate = np.ones(B)
                if cfg.mode == "better_q":
                    # listen to the teacher only where my own outcome fell short of
                    # my critic's prediction (negative prediction error)
                    gate = (ro["adv"] < 0).astype(float)
                qpass_log[k] = float(gate.mean())
            for _ in range(cfg.epochs):
                perm = rng.permutation(B)
                for s0 in range(0, B, mbs):
                    ix = perm[s0:s0 + mbs]
                    o, a, olp = normO[ix], ro["A"][ix], ro["LP"][ix]
                    adv = ro["adv"][ix]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    mu = ag.mu.forward(o)
                    std = np.exp(ag.log_std)
                    lp = logp_gauss(a, mu, ag.log_std)
                    ratio = np.exp(lp - olp)
                    unclipped = ((adv >= 0) & (ratio <= 1 + cfg.clip)) | ((adv < 0) & (ratio >= 1 - cfg.clip))
                    dlp = np.where(unclipped, -ratio * adv, 0.0) / len(ix)
                    z = (a - mu) / std
                    dmu = dlp[:, None] * z / std
                    dls = (dlp[:, None] * (z**2 - 1)).sum(0) - cfg.ent_coef
                    if targets[k] is not None:
                        g = gate[ix][:, None] if cfg.mode == "better_q" else 1.0
                        dmu = dmu + w[k] * g * (mu - targets[k][ix]) / len(ix)
                    ag.mu.step(ag.mu.backward(dmu), lr=lr)
                    ag.step_log_std(dls, lr)
                    v = ag.v.forward(o)[:, 0]
                    ag.v.step(ag.v.backward(cfg.vf_coef * 2 * (v - ro["ret"][ix])[:, None] / len(ix)), lr=lr)

        # ---------------- logging
        step = (upd + 1) * T * N
        log["steps"].append(step)
        for k, b in enumerate(BODIES):
            f = bodies[k].finished[-10:]
            log[f"ret_{b}"].append(float(np.mean(f)) if f else np.nan)
            log[f"w_{b}"].append(w[k])
        log["teacher"].append(teacher)
        log["dist"].append(dist_log)
        log["qpass"].append(qpass_log)
        if (upd + 1) % cfg.eval_every == 0 or upd == n_updates - 1:
            log["eval_steps"].append(step)
            for k, b in enumerate(BODIES):
                r, dx = evaluate(agents[k], b, cfg.eval_eps, cfg.seed, make)
                log[f"eval_{b}"].append(r)
                log[f"dx_{b}"].append(dx)
            print(f"{cfg.mode} s{cfg.seed} {step:>8d}  radial {log['ret_radial'][-1]:7.1f} "
                  f"bilateral {log['ret_bilateral'][-1]:7.1f}  dx {log['dx_radial'][-1]:5.1f}/{log['dx_bilateral'][-1]:5.1f}  teacher {teacher:+d} w {w[0]:.2f}/{w[1]:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------- zero-shot cross-play: each policy driving the other body
    log["cross"] = {f"{BODIES[i]}_policy_on_{BODIES[j]}": evaluate(agents[i], BODIES[j], cfg.eval_eps, cfg.seed, make)
                    for i in range(2) for j in range(2)}
    log["time"] = time.time() - t0
    log["config"] = vars(cfg)
    return log


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="better", choices=["indep", "mutual", "better", "better_q"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--space", default="joint", choices=["joint", "keypoint"],
                   help="joint: same joint tree, joint-space actions (morph.py); "
                        "keypoint: different joint structures, foot-target actions (kp_env.py)")
    p.add_argument("--total-steps", type=int, default=1_000_000, help="per body")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--lam-kl", type=float, default=1.0, help="max distillation weight")
    p.add_argument("--gap-gain", type=float, default=4.0, help="relative gap that saturates the weight = 1/gap_gain")
    p.add_argument("--warmup", type=int, default=10, help="updates before teaching may start")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-eps", type=int, default=3)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    cfg = parse()
    log = train(cfg)
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(log))
    print("cross-play:", json.dumps(log["cross"], indent=1))
