# FedAvg Comparison Analysis

**Date**: 2026-04-14  
**Experiment**: FedAvg vs. Morphic Field on 7×7 maze  
**Config**: seed=42, 60 gens, 20 agents/gen, 100 eps/agent  

## Results Table

| Method | Gen 60 (steps) | Improvement | Displaced Gen 1 | Displaced Adv. |
|--------|---------------:|------------:|----------------:|---------------:|
| FedAvg | 331.2 | 17.2% | 296.8 | 25.8% |
| Morphic (baseline, ρ=0) | 245.2 | 38.7% | 262.4 | 34.4% |
| Morphic (optimal, ρ=0.005) | 262.4 | 34.4% | 228.0 | 43.0% |

## Interpretation

**Control baseline**: 400 steps (agents cannot solve 7×7 maze from scratch in 100 episodes).

### FedAvg vs. Morphic Field (baseline, ρ=0)
- FedAvg Gen 60: **331.2 steps** (17.2% improvement)
- Morphic (ρ=0) Gen 60: **245.2 steps** (38.7% improvement)
- **Gap**: 86.0 steps in favour of morphic field

The performance-weighted morphic field outperforms FedAvg because it suppresses
contributions from failed agents (performance gate: $w_i \approx 0.01$ for unsolved runs).
FedAvg includes these near-zero-quality Q-tables with equal weight, diluting the
accumulated signal.

### Displaced Control Advantage
- FedAvg displaced Gen 1: **296.8 steps** (25.8% advantage)  
- Morphic (ρ=0) displaced Gen 1: **262.4 steps** (34.4% advantage)
- Morphic (ρ=0.005) displaced Gen 1: **228.0 steps** (43.0% advantage)

The morphic field transfers more useful signal to displaced agents because higher-quality
Q-tables dominate the accumulated field.

## Conclusion
Performance-weighted morphic field outperforms FedAvg on both final-generation performance and displaced-control advantage, confirming the paper's positioning claim.

The gap between FedAvg and the morphic field is explained by the performance gate:
FedAvg treats all agents equally, while the morphic field down-weights agents that
failed to solve the maze, concentrating the accumulated Q-table toward successful
trajectory patterns.
