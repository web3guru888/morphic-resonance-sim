#!/usr/bin/env python3
"""
Deep RL Extension — Fixed DQN for morphic field paper.

Improvements over v1:
  1. 5-feature state encoding: [row/H, col/W, (row-goal_r)/H,
                                (col-goal_c)/W, manhattan/(H+W)]
  2. 200 training episodes (was 50)
  3. 32 hidden units (was 16)
  4. morphic field strength 0.5 active from gen 2+

Targets: show DQN can learn on 3x3 maze and morphic field provides advantage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from morphic_sim import Maze, N_ACTIONS, ACTIONS, performance_weight, smooth

SEED = 42
N_GEN = 10
POP = 4
EPISODES = 200       # was 50
MAX_STEPS = 100
MORPHIC_STR = 0.5
HIDDEN = 32          # was 16
STATE_DIM = 5        # was 2


def relu(x):
    return np.maximum(0, x)


class DQNAgent:
    """2-layer numpy DQN with 5-feature state encoding."""

    def __init__(self, maze, hidden=HIDDEN, lr=0.01, gamma=0.95, weights=None):
        self.maze = maze
        self.n_actions = N_ACTIONS
        self.lr = lr
        self.gamma = gamma
        self.state_dim = STATE_DIM
        if weights is not None:
            self.W1 = weights["W1"].copy()
            self.b1 = weights["b1"].copy()
            self.W2 = weights["W2"].copy()
            self.b2 = weights["b2"].copy()
        else:
            self.W1 = np.random.randn(self.state_dim, hidden) * np.sqrt(2.0 / self.state_dim)
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden, self.n_actions) * np.sqrt(2.0 / hidden)
            self.b2 = np.zeros(self.n_actions)
        self.replay = []

    def get_weights(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def encode(self, state):
        r, c = state
        gr, gc = self.maze.goal
        H, W = self.maze.h, self.maze.w
        dist = (abs(r - gr) + abs(c - gc)) / (H + W)
        return np.array([r / H, c / W,
                         (r - gr) / H,
                         (c - gc) / W,
                         dist], dtype=np.float32)

    def forward(self, x):
        z1 = np.clip(x @ self.W1 + self.b1, -10, 10)
        a1 = relu(z1)
        z2 = np.clip(a1 @ self.W2 + self.b2, -10, 10)
        return z2, a1, z1

    def act(self, state_vec, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        q, _, _ = self.forward(state_vec)
        return int(np.argmax(q))

    def store(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))
        if len(self.replay) > 4000:
            self.replay.pop(0)

    def train_batch(self, batch_size=32):
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

        if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(Q2))):
            return

        targets = Q.copy()
        max_q2 = np.max(Q2, axis=1)
        for i in range(batch_size):
            targets[i, A[i]] = R[i] + (1.0 - D[i]) * self.gamma * max_q2[i]

        error = np.clip(Q - targets, -1.0, 1.0)
        dZ2 = error / batch_size
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0).astype(float)
        dW1 = S.T @ dZ1
        db1 = dZ1.sum(0)

        for g in [dW1, db1, dW2, db2]:
            np.nan_to_num(g, copy=False)
            np.clip(g, -1.0, 1.0, out=g)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        for w in [self.W1, self.b1, self.W2, self.b2]:
            np.nan_to_num(w, copy=False)
            np.clip(w, -10, 10, out=w)


class DeepMorphicField:
    """Morphic field over neural network weights (performance-weighted)."""

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
        return {k: np.clip((v / self.weight_sum) * strength, -10, 10)
                for k, v in self.totals.items()}


def _step(maze, state, action):
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if maze.is_open(nr, nc):
        ns = (nr, nc)
        if ns == maze.goal:
            return ns, 1.0, True
        return ns, -0.01, False
    return state, -0.05, False


def train_agent(agent, maze, episodes=EPISODES, max_steps=MAX_STEPS):
    for ep in range(episodes):
        eps = max(0.05, 0.8 - 0.75 * ep / max(episodes - 1, 1))
        state = maze.start
        s_vec = agent.encode(state)
        for _ in range(max_steps):
            a = agent.act(s_vec, eps=eps)
            ns, reward, done = _step(maze, state, a)
            ns_vec = agent.encode(ns)
            agent.store(s_vec, a, reward, ns_vec, float(done))
            if len(agent.replay) >= 32:
                agent.train_batch(32)
            state, s_vec = ns, ns_vec
            if done:
                break


def evaluate_agent(agent, maze, max_steps=MAX_STEPS):
    state = maze.start
    visited = set()
    for t in range(1, max_steps + 1):
        s_vec = agent.encode(state)
        q, _, _ = agent.forward(s_vec)
        a = int(np.argmax(q))
        dr, dc = ACTIONS[a]
        nr, nc = state[0] + dr, state[1] + dc
        if maze.is_open(nr, nc):
            state = (nr, nc)
        if state == maze.goal:
            return t
        # Break cycles
        if state in visited:
            return max_steps
        visited.add(state)
    return max_steps


def run_deep_condition(maze, use_morphic=False, n_gen=N_GEN, pop=POP,
                       episodes=EPISODES, morphic_strength=MORPHIC_STR):
    template = DQNAgent(maze).get_weights()
    field = DeepMorphicField(template) if use_morphic else None
    history = []
    for gen in range(n_gen):
        agents = []
        for _ in range(pop):
            init_w = field.emit(strength=morphic_strength) if (field and field.count > 0) else None
            agent = DQNAgent(maze, weights=init_w)
            train_agent(agent, maze, episodes=episodes)
            agents.append(agent)
        evals = [evaluate_agent(a, maze) for a in agents]
        history.append(np.mean(evals))
        if field is not None:
            for a, ev in zip(agents, evals):
                w = performance_weight(ev, MAX_STEPS)
                field.absorb(a.get_weights(), perf_weight=w)
        print(f"    gen {gen+1:2d}/{n_gen}: {np.mean(evals):.0f} steps "
              f"(min={min(evals):.0f})", flush=True)
    return history, field


def main():
    maze = Maze(width=3, height=3, seed=SEED)
    print(f"Deep RL (v2) Morphic Field Experiment")
    print(f"Maze: {maze.cell_w}×{maze.cell_h}  |  Start {maze.start}  |  Goal {maze.goal}")
    print(f"Config: {N_GEN} gens, {POP} agents/gen, {EPISODES} eps, "
          f"{HIDDEN} hidden, {STATE_DIM}-feature state\n")
    print(maze.render())
    print()

    print("Running Control (no morphic)...")
    np.random.seed(SEED); random.seed(SEED)
    ctrl_hist, _ = run_deep_condition(maze, use_morphic=False)

    print("\nRunning Morphic...")
    np.random.seed(SEED); random.seed(SEED)
    morph_hist, field = run_deep_condition(maze, use_morphic=True)

    print("\nRunning Displaced Control...")
    np.random.seed(SEED + 1000); random.seed(SEED + 1000)
    field_copy = deepcopy(field)
    disp_hist = []
    for gen in range(5):
        agents = []
        for _ in range(POP):
            init_w = field_copy.emit(strength=MORPHIC_STR)
            agent = DQNAgent(maze, weights=init_w)
            train_agent(agent, maze, episodes=EPISODES)
            agents.append(agent)
        evals = [evaluate_agent(a, maze) for a in agents]
        disp_hist.append(np.mean(evals))
        for a, ev in zip(agents, evals):
            w = performance_weight(ev, MAX_STEPS)
            field_copy.absorb(a.get_weights(), perf_weight=w)
        print(f"    gen {gen+1:2d}/5: {np.mean(evals):.0f} steps", flush=True)

    ctrl_avg = np.mean(ctrl_hist)
    morph_final = morph_hist[-1]
    disp_g1 = disp_hist[0]
    adv_pct = (ctrl_avg - disp_g1) / ctrl_avg * 100 if ctrl_avg > 0 else 0

    print(f"\n{'='*55}")
    print(f"Control avg:     {ctrl_avg:.1f} steps")
    print(f"Morphic gen{N_GEN}:   {morph_final:.1f} steps  "
          f"({'SOLVED' if morph_final < MAX_STEPS else 'unsolved'})")
    print(f"Displaced gen1:  {disp_g1:.1f} steps  "
          f"({adv_pct:.1f}% advantage)")
    print(f"{'='*55}")

    # Save figure
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "deep_morphic_results.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Deep RL Morphic Field (3×3 Maze, numpy DQN)\n"
        f"{HIDDEN} hidden | {EPISODES} episodes | {STATE_DIM}-feature state",
        fontsize=13, fontweight="bold"
    )
    sw = min(3, max(len(ctrl_hist) - 1, 1))
    ax.plot(smooth(ctrl_hist, sw), color="#888", linewidth=2.5, ls="--",
            label=f"DQN Control (avg {ctrl_avg:.0f} steps)")
    ax.plot(smooth(morph_hist, sw), color="#FF5722", linewidth=2.5,
            label=f"DQN Morphic (gen{N_GEN}: {morph_final:.0f} steps)")
    disp_x = list(range(N_GEN, N_GEN + len(disp_hist)))
    ax.plot(disp_x, disp_hist, 'o-', color="#9C27B0", linewidth=2.5, markersize=7,
            label=f"Displaced Control (gen1: {disp_g1:.0f} steps, {adv_pct:.0f}% adv.)")
    ax.axhline(ctrl_avg, color="#888", ls=":", alpha=0.5)
    ax.axhline(MAX_STEPS, color="#ccc", ls="--", alpha=0.4, label=f"Max steps ({MAX_STEPS})")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Avg Steps (lower = better)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, MAX_STEPS + 15)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {fig_path}")
    plt.close()

    return ctrl_hist, morph_hist, disp_hist, ctrl_avg, morph_final, disp_g1, adv_pct


if __name__ == "__main__":
    main()
