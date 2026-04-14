#!/usr/bin/env python3
"""
Multi-seed statistical analysis for morphic resonance simulation.
Runs all conditions across seeds 0..29 (30 seeds total), computes
mean ± 95% CI for final-generation performance.

Uses multiprocessing.Pool for parallel execution across seeds.

Usage:
    python multi_seed_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import random
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy import stats

from morphic_sim import Maze, run_condition, run_displaced

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = list(range(30))          # seeds 0..29
N_GEN = 60
POP = 20
EPISODES = 100
MORPHIC_STR = 0.5
MAZE_SEED = 42                   # maze topology is fixed

CONDITIONS = [
    ("Control",            "control"),
    ("Genetic Only",       "genetic"),
    ("Morphic Field Only", "morphic"),
    ("Genetic + Morphic",  "both"),
]

N_WORKERS = min(cpu_count(), len(SEEDS))


# ---------------------------------------------------------------------------
# Per-seed worker (runs in subprocess)
# ---------------------------------------------------------------------------

def run_one_seed(seed):
    """Run all conditions + displaced for a single seed. Returns dict of results."""
    import numpy as np
    import random as rnd
    # Local import to avoid pickling issues with module-level state
    from morphic_sim import Maze, run_condition, run_displaced

    maze = Maze(width=7, height=7, seed=MAZE_SEED)
    seed_results = {}

    for label, cond in CONDITIONS:
        np.random.seed(seed)
        rnd.seed(seed)
        hist, field = run_condition(
            maze, cond,
            n_gen=N_GEN, pop=POP, episodes=EPISODES,
            morphic_strength=MORPHIC_STR
        )
        gen1 = hist[0]
        gen60 = hist[-1]
        imp = (gen1 - gen60) / gen1 * 100 if gen1 > 0 else 0.0
        seed_results[label] = (gen1, gen60, imp)

    # Displaced control: build mature field with 'seed', displaced with seed+1000
    np.random.seed(seed)
    rnd.seed(seed)
    _, mature_field = run_condition(
        maze, "morphic",
        n_gen=N_GEN, pop=POP, episodes=EPISODES,
        morphic_strength=MORPHIC_STR
    )

    np.random.seed(seed + 1000)
    rnd.seed(seed + 1000)
    disp_hist = run_displaced(
        maze, mature_field,
        n_gen=20, pop=POP, episodes=EPISODES,
        morphic_strength=MORPHIC_STR
    )
    seed_results["_displaced_gen1"] = disp_hist[0]

    return seed, seed_results


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(values):
    """Return mean, std, 95% CI half-width (t-distribution)."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    se   = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci = float(t_crit * se)
    return mean, std, ci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("Multi-Seed Morphic Resonance Analysis")
    print(f"Seeds: {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)} total)")
    print(f"Config: {N_GEN} gens, {POP} agents/gen, {EPISODES} episodes")
    print(f"Workers: {N_WORKERS} parallel processes")
    print("=" * 60, flush=True)

    # ---------------------------------------------------------------- Run all
    all_results = {}   # seed -> seed_results dict

    with Pool(processes=N_WORKERS) as pool:
        for seed, seed_res in pool.imap_unordered(run_one_seed, SEEDS):
            all_results[seed] = seed_res
            elapsed = time.time() - t0
            done = len(all_results)
            eta = elapsed / done * (len(SEEDS) - done) if done > 0 else 0
            disp = seed_res["_displaced_gen1"]
            morph_g60 = seed_res["Morphic Field Only"][1]
            print(f"  Seed {seed:02d} done  morphic_g60={morph_g60:5.0f}  "
                  f"displaced_g1={disp:5.0f}  "
                  f"[{done}/{len(SEEDS)}]  ETA {eta/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nAll seeds done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # ---------------------------------------------------------------- Aggregate
    # results[label] → list of (gen1, gen60, improvement%) across seeds
    results_by_cond = {label: [] for label, _ in CONDITIONS}
    displaced_gen1_list = []

    for seed in sorted(all_results.keys()):
        sr = all_results[seed]
        for label, _ in CONDITIONS:
            results_by_cond[label].append(sr[label])
        displaced_gen1_list.append(sr["_displaced_gen1"])

    # ---------------------------------------------------------------- Table
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY  (30 seeds, 95% CI via t-distribution)")
    print("=" * 80)
    header = (f"{'Condition':<26}  {'Gen 1':>8}  {'Gen 60 mean':>12}  "
              f"{'± 95% CI':>9}  {'Improvement':>12}")
    print(header)
    print("-" * 80)

    rows = []
    for label, _ in CONDITIONS:
        vals = results_by_cond[label]
        gen1_vals  = [v[0] for v in vals]
        gen60_vals = [v[1] for v in vals]
        imp_vals   = [v[2] for v in vals]

        g1_mean, g1_std, g1_ci    = compute_stats(gen1_vals)
        g60_mean, g60_std, g60_ci = compute_stats(gen60_vals)
        imp_mean, imp_std, imp_ci = compute_stats(imp_vals)

        row_str = (f"{label:<26}  {g1_mean:>8.1f}  {g60_mean:>12.1f}  "
                   f"±{g60_ci:>8.1f}  {imp_mean:>10.1f}%")
        print(row_str, flush=True)
        rows.append({
            "condition":         label,
            "gen1_mean":         f"{g1_mean:.2f}",
            "gen1_ci":           f"{g1_ci:.2f}",
            "gen60_mean":        f"{g60_mean:.2f}",
            "gen60_std":         f"{g60_std:.2f}",
            "gen60_ci":          f"{g60_ci:.2f}",
            "improvement_mean":  f"{imp_mean:.2f}",
            "improvement_ci":    f"{imp_ci:.2f}",
        })

    # Displaced row
    d_mean, d_std, d_ci = compute_stats(displaced_gen1_list)
    print("-" * 80)
    disp_row_str = (f"{'Displaced Ctrl (Gen 1)':<26}  {'---':>8}  "
                    f"{d_mean:>12.1f}  ±{d_ci:>8.1f}  {'---':>11}")
    print(disp_row_str, flush=True)
    rows.append({
        "condition":        "Displaced Control (Gen 1)",
        "gen1_mean":        "---",
        "gen1_ci":          "---",
        "gen60_mean":       f"{d_mean:.2f}",
        "gen60_std":        f"{d_std:.2f}",
        "gen60_ci":         f"{d_ci:.2f}",
        "improvement_mean": "---",
        "improvement_ci":   "---",
    })
    print("=" * 80)

    # Headline
    ctrl_gen1_mean = np.mean([v[0] for v in results_by_cond["Control"]])
    advantage_vals = [ctrl_gen1_mean - d for d in displaced_gen1_list]
    adv_mean, adv_std, adv_ci   = compute_stats(advantage_vals)
    adv_pct_vals = [a / ctrl_gen1_mean * 100 for a in advantage_vals]
    adv_pct_mean, _, adv_pct_ci = compute_stats(adv_pct_vals)
    print(f"\n*** HEADLINE: Displaced Control Gen 1 = {d_mean:.1f} ± {d_ci:.1f} steps "
          f"(95% CI across 30 seeds)")
    print(f"*** Advantage over control:  {adv_mean:.1f} ± {adv_ci:.1f} steps  "
          f"({adv_pct_mean:.1f} ± {adv_pct_ci:.1f}%)\n")

    # ---------------------------------------------------------------- CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "multi_seed_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {csv_path}", flush=True)

    # ---------------------------------------------------------------- Figure
    fig_path = os.path.join(out_dir, "multi_seed_results.png")

    # Arrays for box plots
    gen60_arrays = [
        [v[1] for v in results_by_cond["Control"]],
        [v[1] for v in results_by_cond["Genetic Only"]],
        [v[1] for v in results_by_cond["Morphic Field Only"]],
        [v[1] for v in results_by_cond["Genetic + Morphic"]],
        displaced_gen1_list,
    ]
    cond_labels_short = ["Control", "Genetic", "Morphic", "Gen+Morph", "Displaced\nGen1"]
    colors = ["#888888", "#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # CI bar data
    means_bar = [
        float(np.mean([v[1] for v in results_by_cond[l]])) for l, _ in CONDITIONS
    ] + [d_mean]
    cis_bar = [
        compute_stats([v[1] for v in results_by_cond[l]])[2] for l, _ in CONDITIONS
    ] + [d_ci]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Multi-Seed Analysis: Morphic Resonance Simulation\n"
        "(30 seeds, 95% CI via t-distribution)",
        fontsize=14, fontweight="bold"
    )

    # Left: mean ± CI bars
    ax = axes[0]
    x = np.arange(len(means_bar))
    bars = ax.bar(x, means_bar, color=colors, alpha=0.8,
                  edgecolor="k", linewidth=0.8)
    ax.errorbar(x, means_bar, yerr=cis_bar, fmt="none",
                capsize=6, capthick=2, elinewidth=2, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels_short, fontsize=10)
    ax.set_ylabel("Avg Steps to Solve (lower = better)", fontsize=11)
    ax.set_title("Final-Generation Performance ± 95% CI", fontsize=12)
    ax.set_ylim(0, 430)
    ax.axhline(400, color="#888", ls="--", alpha=0.4, label="Control baseline")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, mean, ci in zip(bars, means_bar, cis_bar):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + ci + 8,
                f"{mean:.0f}±{ci:.0f}", ha="center", va="bottom", fontsize=8)

    # Right: box plots
    ax2 = axes[1]
    bp = ax2.boxplot(gen60_arrays, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticklabels(cond_labels_short, fontsize=10)
    ax2.set_ylabel("Avg Steps to Solve", fontsize=11)
    ax2.set_title("Distribution Across 30 Seeds (Box Plot)", fontsize=12)
    ax2.axhline(400, color="#888", ls="--", alpha=0.4)
    ax2.set_ylim(0, 430)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {fig_path}", flush=True)
    plt.close()

    return rows, d_mean, d_ci, displaced_gen1_list, results_by_cond


if __name__ == "__main__":
    main()
