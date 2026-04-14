# Morphic Resonance Simulation Analysis

**Date:** 2026-04-14
**Experiment ID:** morphic-sim-001
**Status:** Complete

## Objective

Test whether Sheldrake's morphic resonance hypothesis can be formalized as a computational model using multi-agent reinforcement learning, and whether the predicted data pattern (the displaced control signature) emerges.

## Methodology

### Experimental Setup

- **Environment:** 7x7 procedurally generated maze (DFS recursive backtracker, seed=42)
- **Agents:** Tabular Q-learning (15x15x4 Q-tables, lr=0.2, gamma=0.95)
- **Population:** 20 agents per generation, 60 generations (20 for displaced control)
- **Training:** 100 episodes per agent, max 400 steps per episode

### Five Conditions

1. **Control** --- No inheritance, each generation starts from zero Q-table
2. **Genetic Only** --- Elite selection (top 30%), Q-table crossover + Gaussian mutation (std=0.05)
3. **Morphic Field Only** --- Shared, performance-weighted Q-table; no genetic channel
4. **Genetic + Morphic** --- Both mechanisms, 50/50 blend
5. **Displaced Control** --- Fresh agents (zero ancestry) initialized from mature morphic field at generation 60

### Morphic Field Design

- **Absorption:** `field.total += Q_agent * performance_weight`
- **Performance weight:** `max_steps / eval_steps` for solvers; `0.01` for failures
- **Emission:** `Q_init = (field.total / weight_sum) * 0.5`

## Results

### Main Conditions (60 generations)

| Condition | Gen 1 | Gen 60 | Improvement |
|---|---|---|---|
| Control | 400.0 | 400.0 | 0% |
| Genetic Only | 400.0 | 56.0 | 86% |
| Morphic Field Only | 400.0 | 245.0 | 39% |
| Genetic + Morphic | 400.0 | 108.0 | 73% |

### Displaced Control (20 generations)

| Metric | Value |
|---|---|
| Generation 1 | 297 steps |
| Generation 20 | 228 steps |
| Immediate advantage over control | 103 steps (26%) |

## Key Findings

### 1. Control Baseline is Firm
Individual Q-learning agents cannot solve this 7x7 maze within 100 episodes starting from zeros. All 60 generations remain at 400 steps. This rules out individual learning as an explanation for improvement in other conditions.

### 2. Genetic Inheritance is Powerful
86% improvement via elite selection with crossover. Converges by ~generation 5. Consistent with evolutionary computation literature.

### 3. Morphic Field Transmits Learning Without Genetics
39% improvement with zero genetic inheritance. The improvement is noisier and slower than genetic-only, which is expected: performance-weighted averaging is a blunter aggregation mechanism than elite selection.

### 4. Displaced Control Shows the Signature
Fresh agents with zero ancestry immediately score 297 steps vs. 400 baseline. This is the critical McDougall signature: unrelated agents benefit from collective habit with no physical inheritance channel. In the simulation, we know why (we built the field). In nature, the question is whether such a field exists.

### 5. Combined Conditions Synergize
Genetic + Morphic (73%) is intermediate, not additive. The 50/50 blend dilutes the genetic signal with the noisier morphic prior in early generations.

## Epistemological Assessment

### What This Demonstrates
- The hypothesis is **formally coherent** --- it can be mathematized and implemented
- The predicted data pattern is **distinctive** --- distinguishable from genetic selection and individual learning
- The displaced control is the **most powerful experimental design** for detecting the effect

### What This Does Not Demonstrate
- That morphic resonance exists in nature
- That information can transmit without a physical channel
- That Sheldrake's qualitative description maps onto this specific formalization

### Critical Epistemic Difference
In the simulation, the morphic field is a **design choice** (a NumPy array). In nature, whether any such field exists is an **empirical question**. We demonstrated the *if*, not the *is*.

## Engineering Applications Identified

1. **Population-level experience replay** --- shared policy buffer weighted by success
2. **Cold-start mitigation** --- 26% immediate advantage for new agents
3. **Stigmergic coordination** --- agents coordinate through shared environment
4. **Connection to GraphPalace** --- same stigmergic principle in embedding space vs. Q-table space

## Limitations

1. Tabular Q-learning on small grid (not deep RL)
2. Single maze topology (seed=42 only)
3. Single random seed for all experiments
4. Artificial agents, not biological organisms
5. One of many possible formalizations of the hypothesis

## Future Work

1. Deep RL extension (neural network agents)
2. Multi-maze generalization testing
3. Embedding-space morphic fields (integration with GraphPalace)
4. Multi-seed statistical analysis with confidence intervals
5. Field decay mechanisms (recent agents contribute more)
6. Comparison with FedAvg and other aggregation schemes
