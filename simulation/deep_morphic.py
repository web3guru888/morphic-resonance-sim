#!/usr/bin/env python3
"""Issue #7: Deep RL extension — neural Q-network with morphic field."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import random
from morphic_sim import Maze, N_ACTIONS, ACTIONS, performance_weight, smooth

SEED = 42
N_GEN = 20
POP = 8
EPISODES = 200
MAX_STEPS = 400
MORPHIC_STR = 0.5


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)


class DQNAgent:
    """Simple 2-layer neural network Q-function (numpy only)."""

    def __init__(self, state_dim, n_actions=4, hidden=64, lr=0.0005,
                 gamma=0.95, weights=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.hidden = hidden

        if weights is not None:
            self.W1 = weights["W1"].copy()
            self.b1 = weights["b1"].copy()
            self.W2 = weights["W2"].copy()
            self.b2 = weights["b2"].copy()
        else:
            # Xavier init
            self.W1 = np.random.randn(state_dim, hidden) * np.sqrt(2.0 / state_dim)
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden, n_actions) * np.sqrt(2.0 / hidden)
            self.b2 = np.zeros(n_actions)

        self.replay = []
        self.replay_cap = 500

    def get_weights(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def encode_state(self, state, maze):
        """Normalized coordinate encoding of (row, col) position."""
        return np.array([state[0] / maze.h, state[1] / maze.w], dtype=np.float64)

    def forward(self, x):
        self.z1 = np.clip(x @ self.W1 + self.b1, -50, 50)
        self.a1 = relu(self.z1)
        self.z2 = np.clip(self.a1 @ self.W2 + self.b2, -50, 50)
        return self.z2  # Q-values

    def act(self, state_vec, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        q = self.forward(state_vec)
        return int(np.argmax(q))

    def train_step(self, batch_size=32):
        if len(self.replay) < batch_size:
            return
        batch = random.sample(self.replay, batch_size)
        for s, a, r, s_next, done in batch:
            q_pred = self.forward(s)
            if np.any(np.isnan(q_pred)):
                continue
            q_target = q_pred.copy()
            if done:
                q_target[a] = r
            else:
                q_next = self.forward(s_next)
                if np.any(np.isnan(q_next)):
                    continue
                q_target[a] = r + self.gamma * np.max(q_next)
            # Backward pass (MSE loss gradient)
            error = q_pred - q_target
            # dL/dz2
            dz2 = error / self.n_actions
            # dL/dW2
            dW2 = self.a1.reshape(-1, 1) @ dz2.reshape(1, -1)
            db2 = dz2
            # dL/da1
            da1 = dz2 @ self.W2.T
            dz1 = da1 * relu_deriv(self.z1)
            dW1 = s.reshape(-1, 1) @ dz1.reshape(1, -1)
            db1 = dz1
            # Gradient clipping
            for g in [dW1, db1, dW2, db2]:
                np.clip(g, -1.0, 1.0, out=g)
            # Update with weight clamping
            self.W1 = np.clip(self.W1 - self.lr * dW1, -5, 5)
            self.b1 = np.clip(self.b1 - self.lr * db1, -5, 5)
            self.W2 = np.clip(self.W2 - self.lr * dW2, -5, 5)
            self.b2 = np.clip(self.b2 - self.lr * db2, -5, 5)

    def store(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))
        if len(self.replay) > self.replay_cap:
            self.replay.pop(0)


class DeepMorphicField:
    """Morphic field over neural network weights."""

    def __init__(self, template_weights, decay_rate=0.0):
        self.totals = {k: np.zeros_like(v) for k, v in template_weights.items()}
        self.weight_sum = 0.0
        self.count = 0
        self.decay_rate = decay_rate

    def absorb(self, weights, perf_weight=1.0):
        if self.decay_rate > 0 and self.weight_sum > 0:
            decay = 1.0 - self.decay_rate
            for k in self.totals:
                self.totals[k] *= decay
            self.weight_sum *= decay
        for k in self.totals:
            self.totals[k] += weights[k] * perf_weight
        self.weight_sum += perf_weight
        self.count += 1

    def emit(self, strength=1.0):
        if self.weight_sum == 0:
            return None
        result = {}
        for k, v in self.totals.items():
            w = (v / self.weight_sum) * strength
            np.clip(w, -5.0, 5.0, out=w)
            result[k] = w
        return result


def train_agent(agent, maze, episodes=EPISODES, max_steps=MAX_STEPS):
    for ep in range(episodes):
        eps = max(0.05, 0.5 - 0.45 * ep / max(episodes - 1, 1))
        state = maze.start
        s_vec = agent.encode_state(state, maze)
        for t in range(max_steps):
            a = agent.act(s_vec, eps=eps)
            dr, dc = ACTIONS[a]
            nr, nc = state[0] + dr, state[1] + dc
            if maze.is_open(nr, nc):
                ns = (nr, nc)
                reward = 100.0 if ns == maze.goal else -1.0
            else:
                ns = state
                reward = -5.0
            ns_vec = agent.encode_state(ns, maze)
            done = ns == maze.goal
            agent.store(s_vec, a, reward, ns_vec, done)
            agent.train_step(batch_size=16)
            state = ns
            s_vec = ns_vec
            if done:
                break


def evaluate_agent(agent, maze, max_steps=MAX_STEPS):
    state = maze.start
    for t in range(1, max_steps + 1):
        s_vec = agent.encode_state(state, maze)
        q = agent.forward(s_vec)
        a = int(np.argmax(q))
        dr, dc = ACTIONS[a]
        nr, nc = state[0] + dr, state[1] + dc
        if maze.is_open(nr, nc):
            state = (nr, nc)
        if state == maze.goal:
            return t
    return max_steps


def run_deep_condition(maze, use_morphic=False, n_gen=N_GEN, pop=POP):
    state_dim = 2
    template = DQNAgent(state_dim).get_weights()
    field = DeepMorphicField(template) if use_morphic else None
    history = []

    for gen in range(n_gen):
        agents = []
        for _ in range(pop):
            init_w = field.emit(strength=MORPHIC_STR) if (field and field.count > 0) else None
            agent = DQNAgent(state_dim, weights=init_w)
            train_agent(agent, maze, episodes=EPISODES)
            agents.append(agent)

        evals = [evaluate_agent(a, maze) for a in agents]
        history.append(np.mean(evals))

        if field is not None:
            for a, ev in zip(agents, evals):
                w = performance_weight(ev, MAX_STEPS)
                field.absorb(a.get_weights(), perf_weight=w)

    return history, field


def main():
    maze = Maze(width=5, height=5, seed=SEED)  # 5x5 for deep RL feasibility
    state_dim = 2  # normalized (row, col) coordinates
    print(f"Deep RL Morphic Field Experiment")
    print(f"Maze: {maze.cell_w}x{maze.cell_h} ({state_dim} state dims)")
    print(f"Config: {N_GEN} gens, {POP} agents/gen, {EPISODES} episodes\n")

    print("Running Deep RL Control...")
    np.random.seed(SEED); random.seed(SEED)
    ctrl_hist, _ = run_deep_condition(maze, use_morphic=False)
    print(f"  Control: gen1={ctrl_hist[0]:.0f}  gen{N_GEN}={ctrl_hist[-1]:.0f}")

    print("Running Deep RL Morphic...")
    np.random.seed(SEED); random.seed(SEED)
    morph_hist, field = run_deep_condition(maze, use_morphic=True)
    print(f"  Morphic: gen1={morph_hist[0]:.0f}  gen{N_GEN}={morph_hist[-1]:.0f}")

    # Displaced control
    print("Running Deep RL Displaced Control...")
    from copy import deepcopy
    np.random.seed(999); random.seed(999)
    disp_hist = []
    field_copy = deepcopy(field)
    for gen in range(10):
        agents = []
        for _ in range(POP):
            init_w = field_copy.emit(strength=MORPHIC_STR)
            agent = DQNAgent(state_dim, weights=init_w)
            train_agent(agent, maze, episodes=EPISODES)
            agents.append(agent)
        evals = [evaluate_agent(a, maze) for a in agents]
        disp_hist.append(np.mean(evals))
        for a, ev in zip(agents, evals):
            w = performance_weight(ev, MAX_STEPS)
            field_copy.absorb(a.get_weights(), perf_weight=w)
    print(f"  Displaced: gen1={disp_hist[0]:.0f}  gen10={disp_hist[-1]:.0f}")

    ctrl_avg = np.mean(ctrl_hist)
    adv = (ctrl_avg - disp_hist[0]) / ctrl_avg * 100 if ctrl_avg > 0 else 0
    print(f"\n  Displaced advantage: {adv:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Deep RL Morphic Field (5x5 Maze, numpy DQN)", fontsize=14, fontweight="bold")
    sw = min(3, len(ctrl_hist)-1)
    ax.plot(smooth(ctrl_hist, sw), color="#888", linewidth=2, ls="--", label="DQN Control")
    ax.plot(smooth(morph_hist, sw), color="#FF5722", linewidth=2, label="DQN Morphic")
    # Displaced on separate x range
    disp_x = list(range(N_GEN, N_GEN + len(disp_hist)))
    ax.plot(disp_x, disp_hist, 'o-', color="#9C27B0", linewidth=2, markersize=5,
            label="DQN Displaced Control")
    ax.axhline(ctrl_avg, color="#888", ls=":", alpha=0.5)
    ax.set_xlabel("Generation"); ax.set_ylabel("Avg Steps (lower=better)")
    ax.legend(); ax.grid(True, alpha=0.2); ax.set_ylim(0, 450)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "..", "data", "results", "deep_morphic_results.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")
    plt.close()

if __name__ == "__main__":
    main()
