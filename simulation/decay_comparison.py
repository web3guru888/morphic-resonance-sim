#!/usr/bin/env python3
"""Issue #2: Temporal decay comparison experiment."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import random
from morphic_sim import Maze, Agent, MorphicField, run_condition, run_displaced, smooth, performance_weight

SEED = 42
N_GEN = 60
POP = 20
EPISODES = 100
MORPHIC_STR = 0.5
DECAY_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]

def main():
    maze = Maze(width=7, height=7, seed=SEED)
    print(f"Maze {maze.cell_w}x{maze.cell_h} | Decay comparison experiment")
    print(f"Config: {N_GEN} gens, {POP} agents/gen, {EPISODES} episodes\n")

    # Run control once
    np.random.seed(SEED); random.seed(SEED)
    ctrl_hist, _ = run_condition(maze, "control", N_GEN, POP, EPISODES)
    ctrl_avg = np.mean(ctrl_hist)

    results = {}
    displaced_results = {}

    for dr in DECAY_RATES:
        print(f"  Running morphic-only with decay_rate={dr:.3f}...")
        np.random.seed(SEED); random.seed(SEED)
        hist, field = run_condition(maze, "morphic", N_GEN, POP, EPISODES,
                                   morphic_strength=MORPHIC_STR, decay_rate=dr)
        results[dr] = hist

        # Build fresh field for displaced test
        np.random.seed(SEED); random.seed(SEED)
        _, mature = run_condition(maze, "morphic", N_GEN, POP, EPISODES,
                                 morphic_strength=MORPHIC_STR, decay_rate=dr)
        np.random.seed(999); random.seed(999)
        disp = run_displaced(maze, mature, n_gen=20, pop=POP,
                             episodes=EPISODES, morphic_strength=MORPHIC_STR)
        displaced_results[dr] = disp

    # Print table
    print(f"\n{'Decay Rate':>12} | {'Morphic Gen60':>14} | {'Displaced Gen1':>14} | {'Advantage':>10}")
    print("-" * 60)
    for dr in DECAY_RATES:
        m60 = results[dr][-1]
        d1 = displaced_results[dr][0]
        adv = (ctrl_avg - d1) / ctrl_avg * 100
        print(f"{dr:>12.3f} | {m60:>14.1f} | {d1:>14.1f} | {adv:>9.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Temporal Decay Comparison", fontsize=14, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(DECAY_RATES)))

    ax1 = axes[0]
    for i, dr in enumerate(DECAY_RATES):
        s = smooth(results[dr], 5)
        ax1.plot(s, label=f"decay={dr}", color=colors[i], linewidth=2)
    ax1.axhline(ctrl_avg, color="#888", ls="--", alpha=0.5, label="Control baseline")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Avg Steps")
    ax1.set_title("Morphic-Only: Effect of Decay Rate"); ax1.legend(fontsize=9)
    ax1.set_ylim(0, 450); ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    for i, dr in enumerate(DECAY_RATES):
        s = smooth(displaced_results[dr], 3)
        ax2.plot(s, label=f"decay={dr}", color=colors[i], linewidth=2)
    ax2.axhline(ctrl_avg, color="#888", ls="--", alpha=0.5, label="Control baseline")
    ax2.set_xlabel("Generation (from gen 60)"); ax2.set_ylabel("Avg Steps")
    ax2.set_title("Displaced Control: Effect of Decay Rate"); ax2.legend(fontsize=9)
    ax2.set_ylim(0, 450); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "..", "data", "results", "morphic_decay_comparison.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")
    plt.close()

if __name__ == "__main__":
    main()
