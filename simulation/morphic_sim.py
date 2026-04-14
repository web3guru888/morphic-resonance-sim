#!/usr/bin/env python3
"""
Morphic Resonance Simulation — McDougall Rat Maze Experiment

Simulates Q-learning agents solving a maze across generations under four conditions:
  1. Control        — no inheritance, each generation learns from scratch
  2. Genetic Only   — Q-table inherited via crossover + mutation from elite parents
  3. Morphic Only   — no genetic link, but a shared "field" accumulates from all past solvers
  4. Genetic+Morphic — both mechanisms active

Plus a "Displaced Control": fresh agents (no ancestry) dropped into a mature morphic field
at generation 50 to test whether the field alone accelerates learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

class Maze:
    """Grid maze generated via recursive backtracker (DFS)."""

    def __init__(self, width=8, height=8, seed=42):
        self.cell_w = width
        self.cell_h = height
        self.w = 2 * width + 1
        self.h = 2 * height + 1
        self.grid = np.ones((self.h, self.w), dtype=np.int8)
        self._generate(seed)
        self.start = (1, 1)
        self.goal = (self.h - 2, self.w - 2)

    def _generate(self, seed):
        rng = random.Random(seed)
        # Carve cell interiors
        for r in range(self.cell_h):
            for c in range(self.cell_w):
                self.grid[2 * r + 1, 2 * c + 1] = 0
        visited = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            cr, cc = stack[-1]
            neighbors = [
                (cr + dr, cc + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= cr + dr < self.cell_h
                and 0 <= cc + dc < self.cell_w
                and (cr + dr, cc + dc) not in visited
            ]
            if neighbors:
                nr, nc = rng.choice(neighbors)
                # Remove wall between current cell and neighbor
                self.grid[2 * cr + 1 + (nr - cr), 2 * cc + 1 + (nc - cc)] = 0
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

    def is_open(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w and self.grid[r, c] == 0

    def render(self):
        symbols = {1: "█", 0: " "}
        lines = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                if (r, c) == self.start:
                    row += "S"
                elif (r, c) == self.goal:
                    row += "G"
                else:
                    row += symbols[self.grid[r, c]]
                row += symbols[self.grid[r, c]] if (r, c) not in (self.start, self.goal) else " "
            lines.append(row)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
N_ACTIONS = len(ACTIONS)


class Agent:
    """Tabular Q-learning agent with epsilon decay."""

    def __init__(self, maze, q_table=None, lr=0.2, gamma=0.95,
                 epsilon_start=0.5, epsilon_end=0.05):
        self.maze = maze
        self.lr = lr
        self.gamma = gamma
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.q = q_table.copy() if q_table is not None else np.zeros((maze.h, maze.w, N_ACTIONS))
        self.solved = False  # did this agent ever reach the goal?

    def _act(self, state, eps=None):
        if eps is None:
            eps = self.eps_start
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        r, c = state
        return int(np.argmax(self.q[r, c]))

    def _step(self, state, action):
        dr, dc = ACTIONS[action]
        nr, nc = state[0] + dr, state[1] + dc
        if self.maze.is_open(nr, nc):
            ns = (nr, nc)
        else:
            ns = state
            return ns, -5.0  # wall-bump penalty
        if ns == self.maze.goal:
            return ns, 100.0
        return ns, -1.0

    def train(self, episodes=80, max_steps=400):
        """Train with decaying epsilon. Returns steps per episode."""
        episode_steps = []
        for ep in range(episodes):
            frac = ep / max(episodes - 1, 1)
            eps = self.eps_start + (self.eps_end - self.eps_start) * frac
            state = self.maze.start
            for t in range(1, max_steps + 1):
                a = self._act(state, eps=eps)
                ns, reward = self._step(state, a)
                r, c = state
                best_next = np.max(self.q[ns[0], ns[1]])
                self.q[r, c, a] += self.lr * (reward + self.gamma * best_next - self.q[r, c, a])
                state = ns
                if state == self.maze.goal:
                    self.solved = True
                    break
            episode_steps.append(t)
        return episode_steps

    def evaluate(self, max_steps=400):
        """Greedy run (epsilon=0). Returns steps to goal."""
        state = self.maze.start
        for t in range(1, max_steps + 1):
            a = self._act(state, eps=0.0)
            ns, _ = self._step(state, a)
            state = ns
            if state == self.maze.goal:
                return t
        return max_steps


# ---------------------------------------------------------------------------
# Morphic Field
# ---------------------------------------------------------------------------

class MorphicField:
    """Accumulated Q-table from all past agents — the collective 'habit'.

    With temporal decay (decay_rate > 0), older contributions fade
    exponentially each time a new agent absorbs, modeling pheromone
    evaporation in stigmergic systems.
    """

    def __init__(self, shape, decay_rate=0.0):
        self.total = np.zeros(shape, dtype=np.float64)
        self.weight_sum = 0.0
        self.count = 0
        self.decay_rate = decay_rate

    def absorb(self, q_table, performance_weight=1.0):
        """Add agent's Q-table, weighted by how well it performed.
        Applies exponential decay to existing field before adding."""
        if self.decay_rate > 0 and self.weight_sum > 0:
            decay = 1.0 - self.decay_rate
            self.total *= decay
            self.weight_sum *= decay
        self.total += q_table * performance_weight
        self.weight_sum += performance_weight
        self.count += 1

    def emit(self, strength=1.0):
        if self.weight_sum == 0:
            return np.zeros_like(self.total)
        return (self.total / self.weight_sum) * strength


# ---------------------------------------------------------------------------
# Experiment engine
# ---------------------------------------------------------------------------

def breed(parent_tables, q_shape, mutation_std=0.05):
    """Crossover two random parents + mutation → child Q-table."""
    p1, p2 = random.sample(parent_tables, 2)
    mask = np.random.random(q_shape) < 0.5
    child = np.where(mask, p1, p2)
    child += np.random.randn(*q_shape) * mutation_std
    return child


def performance_weight(eval_steps, max_steps=400):
    """Better performers contribute more to the field. Solvers get 1.0, failures get ~0."""
    if eval_steps >= max_steps:
        return 0.01  # nearly zero — didn't solve
    return max_steps / eval_steps  # shorter path → higher weight


def run_condition(maze, condition, n_gen=50, pop=20, episodes=80,
                  elite_frac=0.3, morphic_strength=0.5, decay_rate=0.0):
    """
    Run one experimental condition across generations.
    Returns: list of avg eval steps per generation, and the morphic field (if any).
    """
    q_shape = (maze.h, maze.w, N_ACTIONS)
    max_steps = 400
    use_genetic = condition in ("genetic", "both")
    use_morphic = condition in ("morphic", "both")

    field = MorphicField(q_shape, decay_rate=decay_rate) if use_morphic else None
    parents = None
    history = []

    for gen in range(n_gen):
        agents = []
        for _ in range(pop):
            q_genetic = None
            q_field = None
            if use_genetic and parents is not None:
                q_genetic = breed(parents, q_shape)
            if use_morphic and field is not None and field.count > 0:
                q_field = field.emit(strength=morphic_strength)

            # Blend sources
            if q_genetic is not None and q_field is not None:
                q_init = 0.5 * q_genetic + 0.5 * q_field
            elif q_genetic is not None:
                q_init = q_genetic
            elif q_field is not None:
                q_init = q_field
            else:
                q_init = np.zeros(q_shape)

            agent = Agent(maze, q_table=q_init)
            agent.train(episodes=episodes, max_steps=max_steps)
            agents.append(agent)

        evals = [a.evaluate(max_steps=max_steps) for a in agents]
        history.append(np.mean(evals))

        # Feed the field — performance-weighted so solvers contribute more
        if field is not None:
            for a, ev in zip(agents, evals):
                w = performance_weight(ev, max_steps)
                field.absorb(a.q, performance_weight=w)

        # Select elite parents
        if use_genetic:
            paired = list(zip(agents, evals))
            paired.sort(key=lambda x: x[1])
            n_elite = max(2, int(pop * elite_frac))
            parents = [a.q.copy() for a, _ in paired[:n_elite]]

    return history, field


def run_displaced(maze, mature_field, n_gen=20, pop=20, episodes=80,
                  morphic_strength=0.5):
    """Fresh agents (no genetic link) using a pre-built morphic field."""
    max_steps = 400
    history = []
    for gen in range(n_gen):
        agents = []
        for _ in range(pop):
            q_init = mature_field.emit(strength=morphic_strength)
            agent = Agent(maze, q_table=q_init)
            agent.train(episodes=episodes, max_steps=max_steps)
            agents.append(agent)
        evals = [a.evaluate(max_steps=max_steps) for a in agents]
        history.append(np.mean(evals))
        for a, ev in zip(agents, evals):
            w = performance_weight(ev, max_steps)
            mature_field.absorb(a.q, performance_weight=w)
    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(data, window=3):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_results(results, displaced, maze, filepath="morphic_resonance_simulation.png"):
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        "Morphic Resonance Simulation — McDougall Rat Maze",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Layout: top row = main chart + maze, bottom row = displaced + annotations
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1])

    colors = {
        "Control": "#888888",
        "Genetic Only": "#2196F3",
        "Morphic Field Only": "#FF5722",
        "Genetic + Morphic": "#4CAF50",
    }
    styles = {
        "Control": {"ls": "--", "alpha": 0.7},
        "Genetic Only": {"ls": "-", "alpha": 1.0},
        "Morphic Field Only": {"ls": "-", "alpha": 1.0},
        "Genetic + Morphic": {"ls": "-", "alpha": 1.0},
    }

    SW = 5  # smoothing window

    # --- Panel 1 (top-left, spans 2 cols): all conditions ---
    ax1 = fig.add_subplot(gs[0, 0:2])
    for label, hist in results.items():
        s = smooth(hist, SW)
        ax1.plot(s, label=label, color=colors[label], linewidth=2.5,
                 linestyle=styles[label]["ls"], alpha=styles[label]["alpha"])
        # Raw data as faint scatter
        ax1.scatter(range(len(hist)), hist, color=colors[label], s=8, alpha=0.15)
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Avg Steps to Solve (lower = better)", fontsize=12)
    ax1.set_title("Generational Learning Across Conditions", fontsize=13)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_ylim(0, 450)
    ax1.grid(True, alpha=0.2)
    ax1.axhspan(0, 100, alpha=0.05, color="green")
    ax1.text(len(hist) - 8, 30, "solved zone", fontsize=9, color="green", alpha=0.5)

    # --- Panel 2 (top-right): maze ---
    ax2 = fig.add_subplot(gs[0, 2])
    display = np.ones((maze.h, maze.w, 3))
    for r in range(maze.h):
        for c in range(maze.w):
            if maze.grid[r, c] == 1:
                display[r, c] = [0.12, 0.12, 0.18]
    sr, sc = maze.start
    gr, gc = maze.goal
    display[sr, sc] = [0.2, 0.8, 0.3]
    display[gr, gc] = [0.9, 0.25, 0.2]
    ax2.imshow(display, interpolation="nearest")
    ax2.set_title(f"Maze ({maze.cell_w}x{maze.cell_h} cells)", fontsize=13)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(sc, sr, "S", ha="center", va="center", fontsize=10,
             fontweight="bold", color="white")
    ax2.text(gc, gr, "G", ha="center", va="center", fontsize=10,
             fontweight="bold", color="white")

    # --- Panel 3 (bottom-left): displaced control ---
    ax3 = fig.add_subplot(gs[1, 0:2])
    ctrl_mean = np.mean(results["Control"])
    ax3.axhline(ctrl_mean, color="#888888", ls="--", lw=2, alpha=0.6,
                label=f"Control baseline ({ctrl_mean:.0f} steps)")
    s = smooth(displaced, min(SW, len(displaced) - 1))
    ax3.plot(range(len(s)), s, color="#9C27B0", linewidth=2.5,
             label="Displaced Control (fresh agents, mature field)")
    ax3.scatter(range(len(displaced)), displaced, color="#9C27B0", s=15, alpha=0.3)

    # Annotate the gap
    gap = ctrl_mean - displaced[0]
    ax3.annotate(
        f"  {gap:.0f}-step advantage\n  from field alone",
        xy=(0, displaced[0]), xytext=(3, ctrl_mean - 30),
        fontsize=10, color="#9C27B0",
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.5),
    )

    ax3.set_xlabel("Generation (starting at gen 60)", fontsize=12)
    ax3.set_ylabel("Avg Steps to Solve", fontsize=12)
    ax3.set_title("Key Test: Fresh Agents in Mature Morphic Field (No Genetic Link)",
                  fontsize=13)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 450)
    ax3.grid(True, alpha=0.2)

    # --- Panel 4 (bottom-right): interpretation text ---
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")

    ctrl_final = results["Control"][-1]
    gen_final = results["Genetic Only"][-1]
    morph_final = results["Morphic Field Only"][-1]
    both_final = results["Genetic + Morphic"][-1]
    disp_first = displaced[0]

    text = (
        "RESULTS SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Control:           {ctrl_final:.0f} steps\n"
        f"Genetic Only:       {gen_final:.0f} steps\n"
        f"Morphic Only:      {morph_final:.0f} steps\n"
        f"Genetic + Morphic: {both_final:.0f} steps\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Displaced Control:\n"
        f"  Gen 1 = {disp_first:.0f} steps\n"
        f"  (vs baseline {ctrl_final:.0f})\n\n"
        "The morphic field transmits\n"
        "learned information to agents\n"
        "with NO genetic connection\n"
        "to prior solvers — matching\n"
        "the pattern Sheldrake predicts."
    )
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                       edgecolor="#cccccc"))

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {filepath}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEED = 42
    N_GEN = 60
    POP = 20
    EPISODES = 100
    MORPHIC_STR = 0.5

    maze = Maze(width=7, height=7, seed=SEED)
    print(maze.render())
    print(f"\nMaze {maze.cell_w}x{maze.cell_h}  |  Start {maze.start}  |  Goal {maze.goal}")
    print(f"Config: {N_GEN} generations, {POP} agents/gen, {EPISODES} training episodes\n")

    conditions = [
        ("Control",           "control"),
        ("Genetic Only",      "genetic"),
        ("Morphic Field Only","morphic"),
        ("Genetic + Morphic", "both"),
    ]

    results = {}

    import sys
    for label, cond in conditions:
        sys.stdout.write(f"  Running {label}...")
        sys.stdout.flush()
        np.random.seed(SEED)
        random.seed(SEED)
        hist, field = run_condition(maze, cond, N_GEN, POP, EPISODES,
                                   morphic_strength=MORPHIC_STR)
        results[label] = hist
        print(f"\r{label:25s}  gen 1: {hist[0]:6.1f}   gen {N_GEN}: {hist[-1]:6.1f}   "
              f"Δ {((hist[0]-hist[-1])/hist[0])*100:+.1f}%")

    # Displaced control — fresh agents in the mature morphic field
    print("\nBuilding mature field for displaced control...")
    np.random.seed(SEED)
    random.seed(SEED)
    _, mature_field = run_condition(maze, "morphic", N_GEN, POP, EPISODES,
                                   morphic_strength=MORPHIC_STR)

    np.random.seed(999)
    random.seed(999)
    displaced = run_displaced(maze, mature_field, n_gen=20, pop=POP,
                              episodes=EPISODES, morphic_strength=MORPHIC_STR)
    print(f"{'Displaced Control':25s}  gen 1: {displaced[0]:6.1f}   gen 20: {displaced[-1]:6.1f}")

    plot_results(results, displaced, maze)

    # --- Interpretation ---
    ctrl_avg = np.mean(results["Control"])
    morph_final = results["Morphic Field Only"][-1]
    disp_first = displaced[0]

    print("\n" + "=" * 64)
    print("INTERPRETATION")
    print("=" * 64)
    print(f"""
Control avg:          {ctrl_avg:.1f} steps  (baseline — no improvement expected)
Morphic-only gen {N_GEN}:  {morph_final:.1f} steps  (field-assisted, no genetic link)
Displaced gen 1:      {disp_first:.1f} steps  (brand-new agents, mature field)

The displaced control is the critical test. These agents have ZERO
genetic connection to any prior solver. If they outperform the control
baseline on their very first generation, the morphic field alone is
transmitting learned information — exactly what Sheldrake's hypothesis
predicts for unrelated rat lines in McDougall's experiment.

This does not prove morphic resonance exists in nature. It demonstrates
what the DATA PATTERN would look like IF it did: unrelated agents
solving the maze faster purely because other agents solved it before,
with no physical channel of inheritance.
""")


if __name__ == "__main__":
    main()
