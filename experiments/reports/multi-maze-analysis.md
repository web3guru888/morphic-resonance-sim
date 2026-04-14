# Multi-Maze Generalization Analysis

**Date:** 2026-04-14 | **Experiment ID:** multimaze-001 | **Status:** Complete

## Objective
Test whether the morphic field captures general maze-solving strategies or merely memorizes topology-specific Q-values.

## Experiment 1: Same-Size Transfer (Field from Maze 42 → Others)

| Maze Seed | Type | Control Avg | Displaced Gen 1 | Advantage |
|---|---|---|---|---|
| 42 | TRAIN | 400 | 280 | 30.1% |
| 123 | TRANSFER | 400 | 400 | 0.0% |
| 456 | TRANSFER | 214 | 36 | **83.2%** |
| 789 | TRANSFER | 400 | 400 | 0.0% |

## Experiment 2: Cross-Maze Field (All 4 mazes → Each + Holdout)

| Maze Seed | Type | Control Avg | Displaced Gen 1 | Advantage |
|---|---|---|---|---|
| 42 | TRAIN | 400 | 280 | 30.1% |
| 123 | TRAIN | 400 | 400 | 0.0% |
| 456 | TRAIN | 214 | 36 | 83.2% |
| 789 | TRAIN | 400 | 400 | 0.0% |
| 999 | **HOLDOUT** | 396 | 365 | **7.9%** |

## Key Findings

1. **Transfer is topology-dependent.** The field from Maze 42 transfers dramatically to Maze 456 (83.2% advantage) but not to Mazes 123 or 789. This suggests the field captures specific navigational patterns (e.g., "move toward bottom-right") that happen to be useful in some topologies but not others.

2. **Cross-maze field shows holdout benefit.** The multi-maze field gives a 7.9% advantage on a completely unseen maze (seed=999). This is modest but positive — the field does capture some general maze-solving heuristic beyond topology-specific Q-values.

3. **Maze difficulty matters.** Maze 456 has a low control baseline (214 steps), meaning agents can partially solve it even without help. The morphic field dramatically accelerates this — suggesting the field is most useful when the task is learnable but difficult.

4. **Some mazes are resistant to transfer.** Mazes 123 and 789 show zero transfer benefit in both experiments. The Q-values from other mazes may map to walls in these topologies, canceling any benefit.

## Implications
The morphic field is not a universal prior — it's a task-specific knowledge accumulator that transfers best to structurally similar environments. For practical multi-agent systems, this means morphic fields should be maintained per-domain rather than globally.

## Parameters
40 generations, 20 agents/gen, 100 episodes, seed=42, morphic strength=0.5. All mazes 7x7.
