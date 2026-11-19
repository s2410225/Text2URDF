"""Dual-policy learning: two policies trained together, learning from their difference.

Inspired by predictive processing in the brain: the brain predicts what it is
about to do (efference copy / habit), compares that prediction with what is
actually done and what happens next, and the mismatch drives learning.

Everything is numpy only (small tanh MLPs with hand-written backprop and Adam),
so it runs on a CPU-only machine with no deep-learning framework.

Modes
-----
ppo         single PPO agent (baseline)
ppo2        two independent PPO agents, each on half the envs (control: same
            env steps, two policies, no interaction)
mutual      two PPO agents, each also pulled toward the other with
            KL(pi_other || pi_self)  (symmetric "difference" as a teacher)
disagree    two PPO agents; intrinsic reward = beta * JSD(pi_1, pi_2)(s)
            (states where the two policies disagree are "surprising")
pred_reg    actor + predictor. The predictor learns to predict the actor's
            action distribution (slow, like an expectation / habit). The actor
            is PPO + lambda * KL(pi_actor || pi_pred): it learns from reward
            but is held back by the mismatch with its own prediction.
pred_gate   actor + predictor. Each sample's policy-gradient weight is scaled
            by the prediction error (surprise = -log pi_pred(a|s)), so
            transitions the predictor did not expect are learned more strongly.
pred_sil    actor + predictor. The predictor only consolidates transitions
            whose outcome was better than expected (advantage > 0); the actor
            is pulled toward this "good-outcome expectation" by KL, weighted
            by how far the actor's own choice deviates from it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------- numpy MLP


class MLP:
    """tanh MLP with explicit forward/backward and an Adam optimizer."""

    def __init__(self, sizes, rng, out_scale=1.0, lr=3e-4):
        self.params = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_scale if i == len(sizes) - 2 else np.sqrt(2)
            w = rng.standard_normal((a, b))
            q, _ = np.linalg.qr(w if a >= b else w.T)
            w = (q if a >= b else q.T)[:a, :b] * gain  # orthogonal init
            self.params += [w.astype(np.float64), np.zeros(b)]
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x):
        self.cache = [x]
        h = x
        n = len(self.params) // 2
        for i in range(n):
            h = h @ self.params[2 * i] + self.params[2 * i + 1]
            if i < n - 1:
                h = np.tanh(h)
            self.cache.append(h)
        return h

    def backward(self, dout):
        grads = [None] * len(self.params)
        n = len(self.params) // 2
        d = dout
        for i in reversed(range(n)):
            if i < n - 1:
                d = d * (1 - self.cache[i + 1] ** 2)
            grads[2 * i] = self.cache[i].T @ d
            grads[2 * i + 1] = d.sum(0)
            d = d @ self.params[2 * i].T
        return grads

    def step(self, grads, max_norm=0.5, lr=None):
        lr = self.lr if lr is None else lr
        norm = np.sqrt(sum((g**2).sum() for g in grads))
        if norm > max_norm:
            grads = [g * (max_norm / (norm + 1e-8)) for g in grads]
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i, g in enumerate(grads):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1**self.t)
            vh = self.v[i] / (1 - b2**self.t)
            self.params[i] -= lr * mh / (np.sqrt(vh) + 1e-5)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def kl(logp, logq):
    """KL(p||q) per row, from log-probs."""
    return (np.exp(logp) * (logp - logq)).sum(-1)


def dkl_dlogits_p(logp, logq):
    """d KL(p||q) / d logits_p (q fixed)."""
    p = np.exp(logp)
    k = (p * (logp - logq)).sum(-1, keepdims=True)
    return p * ((logp - logq) - k)


def dkl_dlogits_q(logp, logq):
    """d KL(p||q) / d logits_q (p fixed) = q - p."""
    return np.exp(logq) - np.exp(logp)


# ---------------------------------------------------------------- agent


class Agent:
    def __init__(self, obs_dim, n_act, rng, hidden=64, lr=3e-4):
        self.pi = MLP([obs_dim, hidden, hidden, n_act], rng, out_scale=0.01, lr=lr)
        self.v = MLP([obs_dim, hidden, hidden, 1], rng, out_scale=1.0, lr=lr)
        self.n_act = n_act

    def logp(self, obs):
        return log_softmax(self.pi.forward(obs))

    def act(self, obs, rng, greedy=False):
        lp = self.logp(obs)
        if greedy:
            return lp.argmax(-1), lp
        p = np.exp(lp)
        u = rng.random((len(p), 1))
        a = (p.cumsum(-1) < u).sum(-1)
        return np.minimum(a, self.n_act - 1), lp

    def value(self, obs):
        return self.v.forward(obs)[:, 0]


class Predictor:
    """A second policy that predicts which action the actor will take."""

    def __init__(self, obs_dim, n_act, rng, hidden=64, lr=3e-4):
        self.pi = MLP([obs_dim, hidden, hidden, n_act], rng, out_scale=0.01, lr=lr)

    def logp(self, obs):
        return log_softmax(self.pi.forward(obs))


# ---------------------------------------------------------------- env helpers


class Envs:
    """Minimal vector env with manual auto-reset; records finished episodes."""

    def __init__(self, env_id, n, seed):
        self.envs = [gym.make(env_id) for _ in range(n)]
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.ep_ret = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=int)
        self.finished = []  # (ep_return)
        # running std of the discounted return, for reward scaling
        self.disc = np.zeros(n)
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def scale(self, r, dones, gamma):
        self.disc = self.disc * gamma + r
        b_mean, b_var, b_n = self.disc.mean(), self.disc.var(), len(r)
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot
        self.disc[dones > 0] = 0.0
        return np.clip(r / np.sqrt(self.var + 1e-8), -10, 10)

    def step(self, actions):
        rews, dones = np.zeros(len(self.envs)), np.zeros(len(self.envs))
        nxt = []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = e.step(int(a))
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            if term or trunc:
                self.finished.append(self.ep_ret[i])
                self.ep_ret[i], self.ep_len[i] = 0, 0
                o, _ = e.reset()
                dones[i] = 1
            rews[i] = r
            nxt.append(o)
        self.obs = np.stack(nxt)
        return rews, dones


def evaluate(policy_logp, env_id, episodes, seed):
    env = gym.make(env_id)
    rets = []
    for k in range(episodes):
        o, _ = env.reset(seed=10_000 + seed * 100 + k)
        done, R = False, 0.0
        while not done:
            a = int(policy_logp(o[None].astype(np.float64)).argmax(-1)[0])
            o, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        rets.append(R)
    return float(np.mean(rets))


def gae(rews, vals, dones, last_val, gamma, lam):
    T = len(rews)
    adv = np.zeros_like(rews)
    last = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else vals[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rews[t] + gamma * nv * nonterm - vals[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + vals


# ---------------------------------------------------------------- training


def ppo_policy_grad(logp_all, act, old_logp_a, adv, clip, ent_coef, weight=None):
    """Gradient of the PPO loss wrt logits (mean over batch). Returns (dlogits, stats)."""
    B = len(act)
    idx = np.arange(B)
    logp_a = logp_all[idx, act]
    ratio = np.exp(logp_a - old_logp_a)
    unclipped = ((adv >= 0) & (ratio <= 1 + clip)) | ((adv < 0) & (ratio >= 1 - clip))
    w = np.ones(B) if weight is None else weight
    # d(-r*A)/dlogp_a = -r*A where the unclipped branch is active
    dlogp_a = np.where(unclipped, -ratio * adv * w, 0.0)
    onehot = np.zeros_like(logp_all)
    onehot[idx, act] = 1
    p = np.exp(logp_all)
    d = dlogp_a[:, None] * (onehot - p)
    # entropy bonus: loss -= ent_coef * H
    H = -(p * logp_all).sum(-1, keepdims=True)
    d += ent_coef * p * (logp_all + H)
    return d / B, {"clipfrac": float((~unclipped).mean()), "entropy": float(H.mean())}


def train(cfg):
    rng = np.random.default_rng(cfg.seed)
    probe = gym.make(cfg.env)
    obs_dim = probe.observation_space.shape[0]
    n_act = probe.action_space.n
    mode = cfg.mode
    two_actors = mode in ("ppo2", "mutual", "disagree")
    n_agents = 2 if two_actors else 1
    agents = [Agent(obs_dim, n_act, rng, lr=cfg.lr) for _ in range(n_agents)]
    pred = Predictor(obs_dim, n_act, rng, lr=cfg.lr * cfg.pred_lr_mult) if mode.startswith("pred") else None

    per = cfg.num_envs // n_agents
    envs = [Envs(cfg.env, per, cfg.seed * 1000 + k * 100) for k in range(n_agents)]
    T = cfg.num_steps
    batch_per_update = cfg.num_envs * T
    n_updates = cfg.total_steps // batch_per_update
    log = {"steps": [], "ret": [], "pe": [], "kl12": [], "eval": [], "eval_pred": [], "eval_steps": []}
    t0 = time.time()
    global_step = 0

    for upd in range(n_updates):
        frac = 1.0 - upd / n_updates
        lr_now = cfg.lr * frac
        beta = cfg.beta * frac
        rollouts = []
        for k, (ag, env) in enumerate(zip(agents, envs)):
            O = np.zeros((T, per, obs_dim))
            A = np.zeros((T, per), dtype=int)
            LP = np.zeros((T, per))
            R = np.zeros((T, per))
            D = np.zeros((T, per))
            V = np.zeros((T, per))
            for t in range(T):
                o = env.obs.astype(np.float64)
                a, lp = ag.act(o, rng)
                O[t], A[t], LP[t] = o, a, lp[np.arange(per), a]
                V[t] = ag.value(o)
                r, d = env.step(a)
                if cfg.norm_reward:
                    r = env.scale(r, d, cfg.gamma)
                R[t], D[t] = r, d
            last_v = ag.value(env.obs.astype(np.float64))
            rollouts.append(dict(O=O, A=A, LP=LP, R=R, D=D, V=V, last_v=last_v))
        global_step += batch_per_update

        # ---- intrinsic reward from disagreement between the two policies
        if mode == "disagree":
            for ro in rollouts:
                flat = ro["O"].reshape(-1, obs_dim)
                p1, p2 = np.exp(agents[0].logp(flat)), np.exp(agents[1].logp(flat))
                m = 0.5 * (p1 + p2)
                jsd = 0.5 * (p1 * np.log(p1 / m + 1e-12)).sum(-1) + 0.5 * (p2 * np.log(p2 / m + 1e-12)).sum(-1)
                ro["R"] = ro["R"] + beta * (jsd / np.log(2)).reshape(T, per)

        for ro in rollouts:
            adv, ret = gae(ro["R"], ro["V"], ro["D"], ro["last_v"], cfg.gamma, cfg.lam)
            ro["adv"], ro["ret"] = adv.reshape(-1), ret.reshape(-1)
            ro["O"] = ro["O"].reshape(-1, obs_dim)
            ro["A"] = ro["A"].reshape(-1)
            ro["LP"] = ro["LP"].reshape(-1)

        # ---- prediction error (actor vs predictor) on this batch, for logging/gating
        pe_mean = np.nan
        if pred is not None:
            ro = rollouts[0]
            plp = pred.logp(ro["O"])
            surprise = -plp[np.arange(len(ro["A"])), ro["A"]]
            ro["surprise"] = surprise
            pe_mean = float(kl(agents[0].logp(ro["O"]), plp).mean())

        # ---- updates
        for k, (ag, ro) in enumerate(zip(agents, rollouts)):
            N = len(ro["A"])
            mb = N // cfg.minibatches
            for _ in range(cfg.epochs):
                perm = rng.permutation(N)
                for s in range(0, N, mb):
                    ix = perm[s : s + mb]
                    o, a, olp = ro["O"][ix], ro["A"][ix], ro["LP"][ix]
                    adv = ro["adv"][ix]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    weight = None
                    if mode == "pred_gate":
                        sp = ro["surprise"][ix]
                        weight = 1.0 + cfg.kappa * (sp / (ro["surprise"].mean() + 1e-8) - 1.0)
                        weight = np.clip(weight, 0.25, 4.0)
                    logits = ag.pi.forward(o)
                    lp_all = log_softmax(logits)
                    d, _ = ppo_policy_grad(lp_all, a, olp, adv, cfg.clip, cfg.ent_coef, weight)

                    if mode == "mutual":
                        other = agents[1 - k]
                        lq = other.logp(o)  # teacher = the other policy (fixed)
                        # loss += lam * KL(other || self): grad wrt self logits = p_self - p_other
                        d += cfg.lam_kl * dkl_dlogits_q(lq, lp_all) / len(ix)
                        ag.pi.forward(o)  # restore cache for backward
                    elif mode in ("pred_reg", "pred_sil"):
                        lq = pred.logp(o)
                        g = dkl_dlogits_p(lp_all, lq)  # loss += lam * KL(actor || pred)
                        if mode == "pred_sil":
                            # the further the actor's choice is from the good-outcome
                            # expectation, the harder it is pulled back (gated by PE)
                            pe = np.clip(-lq[np.arange(len(a)), a] / np.log(n_act), 0, 3)[:, None]
                            g = g * pe
                        d += cfg.lam_kl * g / len(ix)
                        ag.pi.forward(o)
                    ag.pi.step(ag.pi.backward(d), lr=lr_now)

                    vpred = ag.v.forward(o)[:, 0]
                    dv = cfg.vf_coef * 2 * (vpred - ro["ret"][ix])[:, None] / len(ix)
                    ag.v.step(ag.v.backward(dv), lr=lr_now)

                    # ---- predictor learns from the mismatch with the actor
                    if pred is not None:
                        target = ag.logp(o)
                        lq = pred.logp(o)
                        if mode == "pred_sil":
                            # consolidate only better-than-expected outcomes:
                            # weighted cross-entropy on the executed action
                            w = (adv > 0).astype(float) * np.minimum(adv, 3.0)
                            onehot = np.zeros_like(lq)
                            onehot[np.arange(len(a)), a] = 1
                            dq = w[:, None] * (np.exp(lq) - onehot) / max(w.sum(), 1e-8)
                        else:
                            dq = dkl_dlogits_q(target, lq) / len(ix)  # min KL(actor || pred)
                        pred.pi.step(pred.pi.backward(dq), lr=lr_now * cfg.pred_lr_mult)

        # ---- logging
        recent = [r for env in envs for r in env.finished[-10:]]
        log["steps"].append(global_step)
        log["ret"].append(float(np.mean(recent)) if recent else np.nan)
        log["pe"].append(pe_mean)
        if n_agents == 2:
            o = np.concatenate([rollouts[0]["O"], rollouts[1]["O"]])
            log["kl12"].append(float(kl(agents[0].logp(o), agents[1].logp(o)).mean()))
        if (upd + 1) % cfg.eval_every == 0 or upd == n_updates - 1:
            ev = np.mean([evaluate(ag.logp, cfg.env, cfg.eval_eps, cfg.seed) for ag in agents])
            log["eval"].append(float(ev))
            log["eval_steps"].append(global_step)
            if pred is not None:
                log["eval_pred"].append(evaluate(pred.logp, cfg.env, cfg.eval_eps, cfg.seed))
            if cfg.verbose:
                print(f"{cfg.env} {mode} s{cfg.seed} step {global_step} ret {log['ret'][-1]:.1f} "
                      f"eval {ev:.1f} pe {pe_mean:.3f} ({time.time() - t0:.0f}s)", flush=True)
    log["time"] = time.time() - t0
    log["config"] = vars(cfg)
    return log


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--mode", default="ppo",
                   choices=["ppo", "ppo2", "mutual", "disagree", "pred_reg", "pred_gate", "pred_sil"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lam-kl", type=float, default=0.5, help="weight of the policy-difference term")
    p.add_argument("--beta", type=float, default=0.1, help="intrinsic reward scale (disagree)")
    p.add_argument("--kappa", type=float, default=0.5, help="surprise gate strength (pred_gate)")
    p.add_argument("--pred-lr-mult", type=float, default=0.5, help="predictor lr relative to actor")
    p.add_argument("--norm-reward", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-eps", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    cfg = parse()
    log = train(cfg)
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(log))
    print(f"done {cfg.env} {cfg.mode} seed {cfg.seed}: final eval {log['eval'][-1]:.1f} in {log['time']:.0f}s")
