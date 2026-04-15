#!/usr/bin/env python3
"""O(1/sqrt(T)) Convergence Analysis — Full Paper Parameters"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from morphic_sim import Maze, run_condition

N_GEN=60; POP=20; EPISODES=100; N_SEEDS=5; MORPH_STR=0.5; MAZE_SEED=42

def run_seed(seed):
    maze = Maze(width=7, height=7, seed=MAZE_SEED)
    np.random.seed(seed)
    h_c, _ = run_condition(maze, "control", N_GEN, POP, EPISODES, 0.0)
    np.random.seed(seed)
    h_m, _ = run_condition(maze, "morphic", N_GEN, POP, EPISODES, MORPH_STR)
    print(f"  seed {seed}: ctrl_final={h_c[-1]:.0f}  morph_final={h_m[-1]:.0f}", flush=True)
    return h_c, h_m

print(f"O(1/sqrt(T)) analysis: {N_SEEDS} seeds x {N_GEN} gens x POP={POP} x EPS={EPISODES}", flush=True)
t0 = time.time()
with Pool(min(4, N_SEEDS)) as pool:
    results = pool.map(run_seed, range(N_SEEDS))
print(f"Done in {time.time()-t0:.0f}s", flush=True)

ctrl_arr  = np.array([r[0] for r in results])
morph_arr = np.array([r[1] for r in results])
T = np.arange(1, N_GEN + 1)
morph_mean = morph_arr.mean(axis=0)
OPTIMAL = morph_mean.min()
regret = morph_mean - OPTIMAL
cum_avg_regret = np.cumsum(regret) / T

print(f"Control mean: {ctrl_arr.mean():.1f}  Morphic final: {morph_mean[-1]:.1f}", flush=True)
print(f"Optimal (min morph): {OPTIMAL:.1f}", flush=True)
print(f"Cum regret T=1:{cum_avg_regret[0]:.1f}  T={N_GEN}:{cum_avg_regret[-1]:.1f}  Reduction:{(1-cum_avg_regret[-1]/cum_avg_regret[0])*100:.0f}%", flush=True)

def inv_sqrt(T, a, c): return a / np.sqrt(T) + c
def inv_T(T, a, c):    return a / T + c
def inv_log(T, a, c):  return a / np.log(T+1) + c

r2s = {}
popts = {}
for name, func in [("sqrt", inv_sqrt), ("T", inv_T), ("log", inv_log)]:
    try:
        popt, _ = curve_fit(func, T, cum_avg_regret, maxfev=10000)
        fitted = func(T, *popt)
        ss_res = np.sum((cum_avg_regret - fitted)**2)
        ss_tot = np.sum((cum_avg_regret - cum_avg_regret.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        r2s[name] = r2; popts[name] = popt
        print(f"  1/{name}: R2={r2:.4f}", flush=True)
    except Exception as e:
        print(f"  1/{name}: FAILED ({e})", flush=True)

r_pearson, p_pearson = pearsonr(1/np.sqrt(T), cum_avg_regret)
print(f"Pearson r(cum_regret, 1/sqrt(T)) = {r_pearson:.4f}, p = {p_pearson:.2e}", flush=True)

T_fine = np.linspace(1, N_GEN, 400)
r2_sqrt = r2s.get("sqrt", 0)
r2_T    = r2s.get("T",    0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
label_sqrt = "O(1/sqrt(T))  R2=" + f"{r2_sqrt:.3f}"
label_T    = "O(1/T)        R2=" + f"{r2_T:.3f}"
fig.suptitle(f"Morphic Field Regret Convergence (POP={POP}, EPS={EPISODES}, {N_SEEDS} seeds, 7x7 maze)", fontsize=11)
ax1.scatter(T, cum_avg_regret, s=25, alpha=0.6, color="steelblue", zorder=5, label="Data (cum. avg. regret)")
if "sqrt" in popts: ax1.plot(T_fine, inv_sqrt(T_fine, *popts["sqrt"]), "r-",  lw=2, label=label_sqrt)
if "T"    in popts: ax1.plot(T_fine, inv_T(T_fine,    *popts["T"]),    "g--", lw=2, label=label_T)
ax1.set_xlabel("Generation T"); ax1.set_ylabel("Cumulative Avg Regret (steps)")
ax1.set_title("A: Morphic Field Regret Convergence"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
r_label = "r = " + f"{r_pearson:.4f}" + ", p = " + f"{p_pearson:.1e}"
ax2.scatter(1/np.sqrt(T), cum_avg_regret, s=25, alpha=0.6, color="steelblue", zorder=5)
if "sqrt" in popts:
    xfit = np.linspace(0, 1.02, 300)
    ax2.plot(xfit, popts["sqrt"][0]*xfit + popts["sqrt"][1], "r-", lw=2, label=r_label)
ax2.set_xlabel("1 / sqrt(T)"); ax2.set_ylabel("Cumulative Avg Regret (steps)")
ax2.set_title("B: Linearisation Check  (r=" + f"{r_pearson:.3f})"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sqrt_T_confirmed.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print("Figure saved -> " + out_path, flush=True)
np.save(os.path.join(out_dir, "sqrt_T_cum_regret.npy"), cum_avg_regret)
np.save(os.path.join(out_dir, "sqrt_T_T_values.npy"), T)
r2_sqrt = r2s.get("sqrt", 0)
if r2_sqrt > 0.7 and r_pearson >= 0.8:
    print("VERDICT: CONFIRMED O(1/sqrt(T))  R2=" + f"{r2_sqrt:.3f}" + "  r=" + f"{r_pearson:.3f}")
elif r2_sqrt > 0.5:
    print("VERDICT: PARTIAL  R2=" + f"{r2_sqrt:.3f}")
else:
    print("VERDICT: NOT CONFIRMED  R2=" + f"{r2_sqrt:.3f}")
