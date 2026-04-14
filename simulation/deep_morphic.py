#!/usr/bin/env python3
"""Issue #7: Deep RL extension — neural Q-network with morphic field.

Demonstrates that the morphic field pattern (absorb/emit/blend) works with
neural network weights, not just tabular Q-tables.  Uses a minimal numpy-only
DQN on a 3x3 maze for tractable runtime.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import random
from morphic_sim import Maze, N_ACTIONS, ACTIONS, performance_weight, smooth

SEED = 42
N_GEN = 10
POP = 4
EPISODES = 50
MAX_STEPS = 100
MORPHIC_STR = 0.5
HIDDEN = 16


def relu(x):
    return np.maximum(0, x)


class DQNAgent:
    """Minimal 2-layer neural Q-network (numpy only, vectorized batch update)."""

    def __init__(self, state_dim=2, n_actions=4, hidden=HIDDEN, lr=0.005,
                 gamma=0.95, weights=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        if weights is not None:
            self.W1 = weights["W1"].copy()
            self.b1 = weights["b1"].copy()
            self.W2 = weights["W2"].copy()
            self.b2 = weights["b2"].copy()
        else:
            self.W1 = np.random.randn(state_dim, hidden) * np.sqrt(2.0 / state_dim)
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden, n_actions) * np.sqrt(2.0 / hidden)
            self.b2 = np.zeros(n_actions)
        self.replay = []

    def get_weights(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    @staticmethod
    def encode(state, maze):
        return np.array([state[0] / max(maze.h, 1), state[1] / max(maze.w, 1)])

    def forward(self, x):
        """x: (state_dim,) or (batch, state_dim)"""
        z1 = np.clip(x @ self.W1 + self.b1, -5, 5)
        a1 = relu(z1)
        z2 = np.clip(a1 @ self.W2 + self.b2, -5, 5)
        return z2, a1, z1

    def act(self, state_vec, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        q, _, _ = self.forward(state_vec)
        return int(np.argmax(q))

    def store(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))
        if len(self.replay) > 2000:
            self.replay.pop(0)

    def train_batch(self, batch_size=32):
        """Vectorized batch update — one forward/backward for the whole batch."""
        if len(self.replay) < batch_size:
            return
        batch = random.sample(self.replay, batch_size)
        S = np.array([b[0] for b in batch])
        A = np.array([b[1] for b in batch])
        R = np.array([b[2] for b in batch])
        S2 = np.array([b[3] for b in batch])
        D = np.array([b[4] for b in batch], dtype=float)

        Q, A1, Z1 = self.forward(S)
        Q2, _, _ = self.forward(S2)

        # Skip if any NaN/inf crept in
        if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(Q2))):
            return

        targets = Q.copy()
        max_q2 = np.max(Q2, axis=1)
        for i in range(batch_size):
            targets[i, A[i]] = R[i] + (1 - D[i]) * self.gamma * max_q2[i]

        error = np.clip(Q - targets, -1.0, 1.0)

        dZ2 = error / batch_size
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0).astype(float)
        dW1 = S.T @ dZ1
        db1 = dZ1.sum(axis=0)

        # Sanitize any NaN/inf from accumulator issues, then clip
        for g in [dW1, db1, dW2, db2]:
            np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(g, -1.0, 1.0, out=g)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Sanitize weights after update
        for w in [self.W1, self.b1, self.W2, self.b2]:
            np.nan_to_num(w, copy=False, nan=0.0, posinf=5.0, neginf=-5.0)
            np.clip(w, -5, 5, out=w)


class DeepMorphicField:
    """Morphic field over neural network weights."""

    def __init__(self, template_weights, decay_rate=0.0):
        self.totals = {k: np.zeros_like(v) for k, v in template_weights.items()}
        self.weight_sum = 0.0
        self.count = 0
        self.decay_rate = decay_rate

    def absorb(self, weights, perf_weight=1.0):
        if self.decay_rate > 0 and self.weight_sum > 0:
            d = 1.0 - self.decay_rate
            for k in self.totals:
                self.totals[k] *= d
            self.weight_sum *= d
        for k in self.totals:
            self.totals[k] += weights[k] * perf_weight
        self.weight_sum += perf_weight
        self.count += 1

    def emit(self, strength=1.0):
        if self.weight_sum == 0:
            return None
        return {k: np.clip((v / self.weight_sum) * strength, -5, 5)
                for k, v in self.totals.items()}


def train_agent(agent, maze, episodes=EPISODES, max_steps=MAX_STEPS):
    for ep in range(episodes):
        eps = max(0.05, 0.5 - 0.45 * ep / max(episodes - 1, 1))
        state = maze.start
        s_vec = agent.encode(state, maze)
        for t in range(max_steps):
            a = agent.act(s_vec, eps=eps)
            dr, dc = ACTIONS[a]
            nr, nc = state[0] + dr, state[1] + dc
            if maze.is_open(nr, nc):
                ns = (nr, nc)
                reward = 1.0 if ns == maze.goal else -0.01
            else:
                ns = state
                reward = -0.05
            ns_vec = agent.encode(ns, maze)
            done = ns == maze.goal
            agent.store(s_vec, a, reward, ns_vec, done)
            if len(agent.replay) >= 32:
                agent.train_batch(32)
            state = ns
            s_vec = ns_vec
            if done:
                break


def evaluate_agent(agent, maze, max_steps=MAX_STEPS):
    state = maze.start
    for t in range(1, max_steps + 1):
        s_vec = agent.encode(state, maze)
        q, _, _ = agent.forward(s_vec)
        a = int(np.argmax(q))
        dr, dc = ACTIONS[a]
        nr, nc = state[0] + dr, state[1] + dc
        if maze.is_open(nr, nc):
            state = (nr, nc)
        if state == maze.goal:
            return t
    return max_steps


def run_deep_condition(maze, use_morphic=False, n_gen=N_GEN, pop=POP):
    template = DQNAgent().get_weights()
    field = DeepMorphicField(template) if use_morphic else None
    history = []
    for gen in range(n_gen):
        agents = []
        for _ in range(pop):
            init_w = field.emit(strength=MORPHIC_STR) if (field and field.count > 0) else None
            agent = DQNAgent(weights=init_w)
            train_agent(agent, maze)
            agents.append(agent)
        evals = [evaluate_agent(a, maze) for a in agents]
        history.append(np.mean(evals))
        if field is not None:
            for a, ev in zip(agents, evals):
                w = performance_weight(ev, MAX_STEPS)
                field.absorb(a.get_weights(), perf_weight=w)
        print(f"    gen {gen+1:2d}/{n_gen}: {np.mean(evals):.0f} steps", flush=True)
    return history, field


def main():
    maze = Maze(width=3, height=3, seed=SEED)  # 3x3 for tractable DQN
    print(f"Deep RL Morphic Field Experiment")
    print(f"Maze: {maze.cell_w}x{maze.cell_h}  |  Start {maze.start}  |  Goal {maze.goal}")
    print(f"Config: {N_GEN} gens, {POP} agents/gen, {EPISODES} episodes, {HIDDEN} hidden\n")

    print("Running Deep RL Control...")
    np.random.seed(SEED); random.seed(SEED)
    ctrl_hist, _ = run_deep_condition(maze, use_morphic=False)

    print("\nRunning Deep RL Morphic...")
    np.random.seed(SEED); random.seed(SEED)
    morph_hist, field = run_deep_condition(maze, use_morphic=True)

    print("\nRunning Deep RL Displaced Control...")
    from copy import deepcopy
    np.random.seed(999); random.seed(999)
    disp_hist = []
    field_copy = deepcopy(field)
    for gen in range(5):
        agents = []
        for _ in range(POP):
            init_w = field_copy.emit(strength=MORPHIC_STR)
            agent = DQNAgent(weights=init_w)
            train_agent(agent, maze)
            agents.append(agent)
        evals = [evaluate_agent(a, maze) for a in agents]
        disp_hist.append(np.mean(evals))
        for a, ev in zip(agents, evals):
            w = performance_weight(ev, MAX_STEPS)
            field_copy.absorb(a.get_weights(), perf_weight=w)
        print(f"    gen {gen+1:2d}/10: {np.mean(evals):.0f} steps", flush=True)

    ctrl_avg = np.mean(ctrl_hist)
    adv = (ctrl_avg - disp_hist[0]) / ctrl_avg * 100 if ctrl_avg > 0 else 0
    print(f"\n{'='*50}")
    print(f"Control avg:     {ctrl_avg:.1f} steps")
    print(f"Morphic gen{N_GEN}:  {morph_hist[-1]:.1f} steps")
    print(f"Displaced gen1:  {disp_hist[0]:.1f} steps")
    print(f"Advantage:       {adv:.1f}%")
    print(f"{'='*50}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Deep RL Morphic Field (3x3 Maze, numpy DQN)", fontsize=14, fontweight="bold")
    sw = min(3, max(len(ctrl_hist)-1, 1))
    ax.plot(smooth(ctrl_hist, sw), color="#888", linewidth=2, ls="--", label="DQN Control")
    ax.plot(smooth(morph_hist, sw), color="#FF5722", linewidth=2, label="DQN Morphic")
    disp_x = list(range(N_GEN, N_GEN + len(disp_hist)))
    ax.plot(disp_x, disp_hist, 'o-', color="#9C27B0", linewidth=2, markersize=5,
            label="DQN Displaced Control")
    ax.axhline(ctrl_avg, color="#888", ls=":", alpha=0.5)
    ax.set_xlabel("Generation"); ax.set_ylabel("Avg Steps (lower=better)")
    ax.legend(); ax.grid(True, alpha=0.2); ax.set_ylim(0, MAX_STEPS + 50)
    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "..", "data", "results", "deep_morphic_results.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")
    plt.close()

if __name__ == "__main__":
    main()
