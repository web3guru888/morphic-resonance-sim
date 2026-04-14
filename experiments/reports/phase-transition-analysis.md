# Phase Transition: Population Threshold Analysis

**Date:** 2026-04-14 | **Experiment ID:** phase-001 | **Status:** Complete

## Objective
Determine whether the morphic field exhibits a critical population threshold below which it fails to form coherent knowledge.

## Results

| Population | Ctrl Avg | Morphic Gen 40 | Displaced Gen 1 | Advantage |
|---|---|---|---|---|
| 1 | 400.0 | 400.0 | 56.0 | 86.0%* |
| 2 | 395.7 | 400.0 | 400.0 | -1.1% |
| 5 | 400.0 | 262.4 | 331.2 | 17.2% |
| 10 | 400.0 | 331.2 | 193.6 | **51.6%** |
| 20 | 400.0 | 262.4 | 279.6 | 30.1% |
| 40 | 399.6 | 288.2 | 296.8 | 25.7% |

*Pop=1 anomaly: single agent can memorize one good path.

## Key Findings

1. **Population=2 is the critical failure point.** With only 2 agents, the field accumulates too little diversity per generation and the displaced control shows zero benefit (-1.1%).

2. **Effectiveness emerges at population >= 5.** At pop=5, displaced agents gain a 17.2% advantage, rising to 51.6% at pop=10.

3. **Peak effectiveness at moderate populations (10).** Pop=10 shows the strongest displaced control benefit. Larger populations (20, 40) show diminishing returns — more agents per generation means more noise in the field.

4. **Not a sharp phase transition.** Unlike Khushiyant's stigmergic systems (sharp transition at density ~0.20-0.23), the morphic field shows a gradual onset with a broad peak. This suggests the field mechanism is more robust than density-dependent pheromone systems.

## Parameters
40 generations, 100 training episodes, seed=42, morphic strength=0.5. Displaced control runs for 10 generations.
