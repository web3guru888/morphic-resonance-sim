#!/usr/bin/env python3
"""Issue #3: Phase transition — critical population threshold for morphic field."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import random
from morphic_sim import Maze, Agent, MorphicField, run_condition, run_displaced, performance_weight

SEED = 42
N_GEN = 40
EPISODES = 100
MORPHIC_STR = 0.5
POPULATIONS = [1, 2, 5, 10, 20, 40]

def main():
    maze = Maze(width=7, height=7, seed=SEED)
    print(f"Phase Transition Experiment | {N_GEN} gens, {EPISODES} episodes")
    print(f"Populations: {POPULATIONS}\n")

    ctrl_results = {}
    morphic_results = {}
    displaced_results = {}

    for pop in POPULATIONS:
        print(f"  Population {pop:3d}...", end=" ", flush=True)

        # Control
        np.random.seed(SEED); random.seed(SEED)
        ctrl_h, _ = run_condition(maze, "control", N_GEN, pop, EPISODES)
        ctrl_results[pop] = ctrl_h

        # Morphic only
        np.random.seed(SEED); random.seed(SEED)
        morph_h, _ = run_condition(maze, "morphic", N_GEN, pop, EPISODES,
                                   morphic_strength=MORPHIC_STR)
        morphic_results[pop] = morph_h

        # Displaced control
        np.random.seed(SEED); random.seed(SEED)
        _, mature = run_condition(maze, "morphic", N_GEN, pop, EPISODES,
                                 morphic_strength=MORPHIC_STR)
        np.random.seed(999); random.seed(999)
        disp = run_displaced(maze, mature, n_gen=10, pop=pop, episodes=EPISODES,
                             morphic_strength=MORPHIC_STR)
        displaced_results[pop] = disp

        ctrl_avg = np.mean(ctrl_h)
        d1 = disp[0]
        adv = (ctrl_avg - d1) / ctrl_avg * 100 if ctrl_avg > 0 else 0
        conv = next((g for g, v in enumerate(morph_h) if v < 350), -1)
        print(f"morphic_final={morph_h[-1]:.0f}  disp_gen1={d1:.0f}  adv={adv:.1f}%  conv_gen={conv}")

    # Print table
    print(f"\n{'Pop':>5} | {'Ctrl Avg':>9} | {'Morph G{}'.format(N_GEN):>10} | {'Disp G1':>8} | {'Advantage':>10} | {'Conv Gen':>9}")
    print("-" * 65)
    for pop in POPULATIONS:
        ca = np.mean(ctrl_results[pop])
        mf = morphic_results[pop][-1]
        d1 = displaced_results[pop][0]
        adv = (ca - d1) / ca * 100 if ca > 0 else 0
        conv = next((g for g, v in enumerate(morphic_results[pop]) if v < 350), -1)
        print(f"{pop:>5} | {ca:>9.1f} | {mf:>10.1f} | {d1:>8.1f} | {adv:>9.1f}% | {conv:>9}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase Transition: Population Threshold for Morphic Field", fontsize=14, fontweight="bold")

    pops = POPULATIONS
    finals = [morphic_results[p][-1] for p in pops]
    ctrl_avgs = [np.mean(ctrl_results[p]) for p in pops]
    disp_g1 = [displaced_results[p][0] for p in pops]
    advantages = [(ca - d) / ca * 100 if ca > 0 else 0 for ca, d in zip(ctrl_avgs, disp_g1)]
    convs = [next((g for g, v in enumerate(morphic_results[p]) if v < 350), N_GEN) for p in pops]

    ax1 = axes[0]
    ax1.plot(pops, finals, 'o-', color="#FF5722", linewidth=2, markersize=8, label="Morphic final")
    ax1.plot(pops, ctrl_avgs, 's--', color="#888", linewidth=1.5, markersize=6, label="Control avg")
    ax1.set_xlabel("Population Size"); ax1.set_ylabel("Avg Steps (lower=better)")
    ax1.set_title("Final Performance vs Population"); ax1.legend()
    ax1.set_xscale("log"); ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.plot(pops, advantages, 'o-', color="#9C27B0", linewidth=2, markersize=8)
    ax2.set_xlabel("Population Size"); ax2.set_ylabel("Advantage (%)")
    ax2.set_title("Displaced Control Advantage vs Population")
    ax2.axhline(0, color="#888", ls="--", alpha=0.5)
    ax2.set_xscale("log"); ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    ax3.plot(pops, convs, 'o-', color="#2196F3", linewidth=2, markersize=8)
    ax3.set_xlabel("Population Size"); ax3.set_ylabel("Gen to reach <350 steps")
    ax3.set_title("Convergence Speed vs Population")
    ax3.set_xscale("log"); ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "..", "data", "results", "phase_transition_results.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")
    plt.close()

if __name__ == "__main__":
    main()
