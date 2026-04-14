#!/usr/bin/env python3
"""Issue #6: Multi-maze generalization — does the field transfer?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import random
from morphic_sim import (Maze, Agent, MorphicField, N_ACTIONS,
                         run_condition, run_displaced, performance_weight, smooth)

SEED = 42
N_GEN = 40
POP = 20
EPISODES = 100
MORPHIC_STR = 0.5
MAZE_SEEDS = [42, 123, 456, 789]
HOLDOUT_SEED = 999


def run_morphic_on_maze(maze, n_gen=N_GEN, pop=POP, episodes=EPISODES):
    """Build a morphic field on a specific maze."""
    np.random.seed(SEED); random.seed(SEED)
    hist, field = run_condition(maze, "morphic", n_gen, pop, episodes,
                                morphic_strength=MORPHIC_STR)
    return hist, field


def test_displaced_on_maze(maze, field, pop=POP, episodes=EPISODES):
    """Test displaced control agents on a maze using a given field."""
    np.random.seed(999); random.seed(999)
    disp = run_displaced(maze, field, n_gen=10, pop=pop, episodes=episodes,
                         morphic_strength=MORPHIC_STR)
    return disp


def experiment1_transfer():
    """Build field on Maze A, test on Mazes B, C, D."""
    print("\n=== Experiment 1: Same-Size Transfer ===")
    train_maze = Maze(width=7, height=7, seed=42)
    _, field = run_morphic_on_maze(train_maze)

    results = {}
    for seed in MAZE_SEEDS:
        test_maze = Maze(width=7, height=7, seed=seed)
        # Control baseline for this maze
        np.random.seed(SEED); random.seed(SEED)
        ctrl, _ = run_condition(test_maze, "control", N_GEN, POP, EPISODES)
        ctrl_avg = np.mean(ctrl)

        from copy import deepcopy
        field_copy = deepcopy(field)
        disp = test_displaced_on_maze(test_maze, field_copy)
        adv = (ctrl_avg - disp[0]) / ctrl_avg * 100 if ctrl_avg > 0 else 0
        results[seed] = {"ctrl_avg": ctrl_avg, "disp_g1": disp[0],
                         "disp_hist": disp, "advantage": adv}
        tag = "TRAIN" if seed == 42 else "TRANSFER"
        print(f"  Maze seed={seed:3d} ({tag:8s}): ctrl={ctrl_avg:.0f}  disp_g1={disp[0]:.0f}  adv={adv:.1f}%")

    return results


def experiment2_crossmaze():
    """Train across multiple mazes, test on each + holdout."""
    print("\n=== Experiment 2: Cross-Maze Field Building ===")
    mazes = {s: Maze(width=7, height=7, seed=s) for s in MAZE_SEEDS}
    q_shape = (mazes[42].h, mazes[42].w, N_ACTIONS)
    field = MorphicField(q_shape)
    max_steps = 400

    # Train across mazes
    for gen in range(N_GEN):
        maze_seed = MAZE_SEEDS[gen % len(MAZE_SEEDS)]
        maze = mazes[maze_seed]
        agents = []
        for _ in range(POP):
            q_init = field.emit(strength=MORPHIC_STR) if field.count > 0 else np.zeros(q_shape)
            agent = Agent(maze, q_table=q_init)
            agent.train(episodes=EPISODES, max_steps=max_steps)
            agents.append(agent)
        evals = [a.evaluate(max_steps=max_steps) for a in agents]
        for a, ev in zip(agents, evals):
            w = performance_weight(ev, max_steps)
            field.absorb(a.q, performance_weight=w)

    # Test on all mazes + holdout
    results = {}
    from copy import deepcopy
    for seed in MAZE_SEEDS + [HOLDOUT_SEED]:
        test_maze = Maze(width=7, height=7, seed=seed)
        np.random.seed(SEED); random.seed(SEED)
        ctrl, _ = run_condition(test_maze, "control", N_GEN, POP, EPISODES)
        ctrl_avg = np.mean(ctrl)

        field_copy = deepcopy(field)
        disp = test_displaced_on_maze(test_maze, field_copy)
        adv = (ctrl_avg - disp[0]) / ctrl_avg * 100 if ctrl_avg > 0 else 0
        tag = "HOLDOUT" if seed == HOLDOUT_SEED else "TRAIN"
        results[seed] = {"ctrl_avg": ctrl_avg, "disp_g1": disp[0], "advantage": adv}
        print(f"  Maze seed={seed:3d} ({tag:7s}): ctrl={ctrl_avg:.0f}  disp_g1={disp[0]:.0f}  adv={adv:.1f}%")

    return results


def main():
    print(f"Multi-Maze Generalization Experiment")
    print(f"Config: {N_GEN} gens, {POP} agents/gen, {EPISODES} episodes")

    exp1 = experiment1_transfer()
    exp2 = experiment2_crossmaze()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Maze Generalization", fontsize=14, fontweight="bold")

    # Exp 1
    ax = axes[0]
    seeds = list(exp1.keys())
    advs = [exp1[s]["advantage"] for s in seeds]
    colors = ["#4CAF50" if s == 42 else "#2196F3" for s in seeds]
    bars = ax.bar([f"seed={s}" for s in seeds], advs, color=colors)
    ax.set_ylabel("Displaced Control Advantage (%)")
    ax.set_title("Exp 1: Field from Maze 42 → Other Mazes")
    ax.axhline(0, color="#888", ls="--"); ax.grid(True, alpha=0.2, axis="y")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)

    # Exp 2
    ax = axes[1]
    seeds2 = list(exp2.keys())
    advs2 = [exp2[s]["advantage"] for s in seeds2]
    colors2 = ["#FF9800" if s == HOLDOUT_SEED else "#4CAF50" for s in seeds2]
    bars2 = ax.bar([f"seed={s}" for s in seeds2], advs2, color=colors2)
    ax.set_ylabel("Displaced Control Advantage (%)")
    ax.set_title("Exp 2: Cross-Maze Field → All Mazes + Holdout")
    ax.axhline(0, color="#888", ls="--"); ax.grid(True, alpha=0.2, axis="y")
    ax.bar_label(bars2, fmt="%.1f%%", fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "..", "data", "results", "multi_maze_results.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")
    plt.close()

if __name__ == "__main__":
    main()
