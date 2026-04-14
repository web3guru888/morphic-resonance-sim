#!/usr/bin/env python3
"""Issue #5: Adversarial robustness — what happens with bad Q-values?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import random
from morphic_sim import (Maze, Agent, MorphicField, N_ACTIONS,
                         performance_weight, smooth)

SEED = 42
N_GEN = 40
POP = 20
EPISODES = 100
MORPHIC_STR = 0.5


class RobustMorphicField(MorphicField):
    """MorphicField with consensus filtering to reject outlier Q-tables."""

    def __init__(self, shape, decay_rate=0.0, consensus_threshold=2.0):
        super().__init__(shape, decay_rate=decay_rate)
        self.running_mean = np.zeros(shape, dtype=np.float64)
        self.running_var = np.ones(shape, dtype=np.float64)
        self.consensus_threshold = consensus_threshold
        self._update_count = 0

    def absorb(self, q_table, performance_weight=1.0):
        """Only absorb Q-values within threshold stds of running mean."""
        if self._update_count > 5:
            std = np.sqrt(self.running_var + 1e-8)
            deviation = np.abs(q_table - self.running_mean) / std
            mean_dev = np.mean(deviation)
            if mean_dev > self.consensus_threshold:
                return  # reject this contribution

        # Update running stats (Welford's)
        self._update_count += 1
        delta = q_table - self.running_mean
        self.running_mean += delta / self._update_count
        delta2 = q_table - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self._update_count

        super().absorb(q_table, performance_weight)


def run_adversarial(maze, field_class, attack_type, n_gen=N_GEN, pop=POP,
                    episodes=EPISODES, morphic_strength=MORPHIC_STR):
    """Run morphic condition with adversarial injection."""
    q_shape = (maze.h, maze.w, N_ACTIONS)
    max_steps = 400
    field = field_class(q_shape)
    history = []
    inject_gen = n_gen // 2  # inject at halfway point

    for gen in range(n_gen):
        agents = []
        for _ in range(pop):
            q_init = field.emit(strength=morphic_strength) if field.count > 0 else np.zeros(q_shape)
            agent = Agent(maze, q_table=q_init)
            agent.train(episodes=episodes, max_steps=max_steps)
            agents.append(agent)

        evals = [a.evaluate(max_steps=max_steps) for a in agents]
        history.append(np.mean(evals))

        # Normal absorption
        for a, ev in zip(agents, evals):
            w = performance_weight(ev, max_steps)
            field.absorb(a.q, performance_weight=w)

        # Adversarial injection at inject_gen
        if gen == inject_gen:
            n_inject = max(1, pop // 2)
            for _ in range(n_inject):
                if attack_type == "random_noise":
                    bad_q = np.random.randn(*q_shape) * 10.0
                    field.absorb(bad_q, performance_weight=max_steps / 50)  # high weight
                elif attack_type == "degenerate":
                    bad_q = np.zeros(q_shape)
                    bad_q[:, :, 2] = 100.0  # always go left
                    field.absorb(bad_q, performance_weight=max_steps / 50)
                elif attack_type == "zero_flood":
                    field.absorb(np.zeros(q_shape), performance_weight=max_steps / 20)

    return history


def main():
    maze = Maze(width=7, height=7, seed=SEED)
    print(f"Adversarial Robustness Experiment | {N_GEN} gens, {POP} agents/gen")

    attacks = ["random_noise", "degenerate", "zero_flood"]
    attack_labels = {"random_noise": "Random Noise Injection",
                     "degenerate": "Degenerate Policy Injection",
                     "zero_flood": "Zero-Flood Attack"}

    results = {}
    for attack in attacks:
        print(f"\n  Attack: {attack_labels[attack]}")
        for label, fc in [("Undefended", MorphicField), ("Robust (filtered)", RobustMorphicField)]:
            np.random.seed(SEED); random.seed(SEED)
            h = run_adversarial(maze, fc, attack)
            results[(attack, label)] = h
            print(f"    {label:25s} pre-attack={np.mean(h[:N_GEN//2]):.0f}  post-attack={np.mean(h[N_GEN//2:]):.0f}")

    # Baseline (no attack)
    np.random.seed(SEED); random.seed(SEED)
    baseline = run_adversarial(maze, MorphicField, "none_placeholder")
    # Remove the attack (re-run cleanly)
    np.random.seed(SEED); random.seed(SEED)
    q_shape = (maze.h, maze.w, N_ACTIONS)
    from morphic_sim import run_condition
    clean_h, _ = run_condition(maze, "morphic", N_GEN, POP, EPISODES, morphic_strength=MORPHIC_STR)

    # Print summary
    print(f"\n{'Attack':>25} | {'Undefended Post':>16} | {'Robust Post':>12} | {'Recovery':>10}")
    print("-" * 75)
    for attack in attacks:
        u_post = np.mean(results[(attack, "Undefended")][N_GEN//2:])
        r_post = np.mean(results[(attack, "Robust (filtered)")][N_GEN//2:])
        recovery = (u_post - r_post) / u_post * 100 if u_post > 0 else 0
        print(f"{attack_labels[attack]:>25} | {u_post:>16.1f} | {r_post:>12.1f} | {recovery:>9.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Adversarial Robustness: Morphic Field Defenses", fontsize=14, fontweight="bold")

    for i, attack in enumerate(attacks):
        ax = axes[i]
        u = results[(attack, "Undefended")]
        r = results[(attack, "Robust (filtered)")]
        sw = min(3, len(u)-1)
        ax.plot(smooth(u, sw), color="#F44336", linewidth=2, label="Undefended")
        ax.plot(smooth(r, sw), color="#4CAF50", linewidth=2, label="Robust (filtered)")
        ax.plot(smooth(clean_h, sw), color="#888", linewidth=1.5, ls="--", label="No attack", alpha=0.6)
        ax.axvline(N_GEN//2, color="red", ls=":", alpha=0.5, label="Attack point")
        ax.set_xlabel("Generation"); ax.set_ylabel("Avg Steps")
        ax.set_title(attack_labels[attack]); ax.legend(fontsize=8)
        ax.set_ylim(0, 450); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "..", "data", "results", "adversarial_robustness.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")
    plt.close()

if __name__ == "__main__":
    main()
