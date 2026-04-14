# Morphic Resonance Simulation

**Collective Habit as Infrastructure: A Computational Model of Morphic Resonance via Multi-Agent Reinforcement Learning**

A computational model of Rupert Sheldrake's morphic resonance hypothesis, implemented as a multi-agent Q-learning simulation of the McDougall rat maze experiment (1920--1954).

## Key Results

| Condition | Gen 1 (steps) | Gen 60 (steps) | Improvement |
|---|---|---|---|
| Control | 400 | 400 | 0% |
| Genetic Only | 400 | 56 | 86% |
| Morphic Field Only | 400 | 245 | 39% |
| Genetic + Morphic | 400 | 108 | 73% |
| **Displaced Control** | **297** | **228** | **26% immediate** |

The **displaced control** is the critical result: fresh agents with zero genetic connection to any prior solver immediately score 297 steps versus the 400-step baseline --- a 26% advantage on their first generation, purely from the accumulated morphic field.

## What This Proves (and Doesn't)

**Does prove:**
- The hypothesis is formally coherent and can be implemented as a mathematical model
- The predicted data pattern is distinctive and detectable (the displaced control signature)
- The displaced control test is a viable experimental design for testing the hypothesis

**Does not prove:**
- That morphic resonance exists in nature (we *built* the field --- it's a data structure)
- That information can transmit without a physical channel
- That Sheldrake's specific mechanism is real

The simulation is a **computational thought experiment** that makes the hypothesis precise enough to test empirically.

## Engineering Applications

Independent of Sheldrake's metaphysics, the morphic field architecture is a practical design pattern:

- **Population-level experience replay** --- performance-weighted shared memory across agent generations
- **Cold-start mitigation** --- new agents bootstrap from collective knowledge (26% immediate advantage)
- **Stigmergic coordination** --- agents coordinate through shared environment, not direct messaging
- **Connection to GraphPalace** --- same principle as pheromone-based stigmergic memory, applied to Q-table space

## Repository Structure

```
morphic-resonance-sim/
├── simulation/              # Source code (Apache 2.0)
│   └── morphic_sim.py       # Main simulation (493 lines Python)
├── paper/                   # Academic paper (CC BY 4.0)
│   ├── morphic-sim-paper.tex
│   ├── morphic-sim-paper.pdf
│   └── Makefile
├── data/                    # Simulation outputs (CC BY 4.0)
│   └── results/
│       └── morphic_resonance_simulation.png
├── experiments/             # Experiment reports
│   └── reports/
│       └── simulation-analysis.md
├── benchmarks/              # Performance benchmarks
│   ├── scripts/
│   └── results/
├── references/              # Background materials
├── LICENSE                  # CC BY 4.0 (paper, docs, data)
├── LICENSE-CODE             # Apache 2.0 (simulation code)
├── CITATION.cff             # Citation metadata
└── README.md
```

## Quick Start

### Requirements

- Python 3.8+
- NumPy
- Matplotlib

### Run the simulation

```bash
cd simulation
python morphic_sim.py
```

This will:
1. Generate a 7x7 maze (seed=42)
2. Run all four experimental conditions across 60 generations
3. Run the displaced control test (20 generations)
4. Save a 4-panel visualization to `morphic_resonance_simulation.png`
5. Print results and interpretation to stdout

### Build the paper

```bash
# Requires pdflatex or tectonic
cd paper
make          # two-pass pdflatex build
# or
tectonic morphic-sim-paper.tex
```

## The Morphic Field Architecture

The core contribution --- a reusable design pattern for multi-agent systems:

```
Absorb:  field.total += agent.Q * performance_weight
Emit:    Q_init = (field.total / field.weight_sum) * strength
Blend:   Q_init = 0.5 * Q_genetic + 0.5 * Q_morphic
```

**Performance weight**: agents that solve the maze quickly contribute proportionally more; failed agents contribute near-zero (0.01). The field does not fill with noise from failed agents.

**Emission strength**: 0.5 --- gives new agents a *nudge* toward past solutions without fully determining behavior.

## Background

### The McDougall Experiment (1920--1954)

William McDougall trained Wistar rats in a water maze at Harvard. Over 30+ generations, errors dropped from ~165 to ~20. The critical finding: Agar and Crew's replication at Edinburgh maintained a *control line* never selected for maze ability --- and the **control line also improved**, suggesting a non-genetic channel of information transfer.

### Sheldrake's Hypothesis

Rupert Sheldrake proposed that natural systems inherit a collective memory through *morphic fields* --- non-material regions of influence that shape form, development, and behavior through *morphic resonance* (similarity-based, non-local, cumulative). Our simulation formalizes this as a shared, performance-weighted Q-table.

## Simulation Parameters

| Parameter | Value |
|---|---|
| Maze size | 7x7 cells (15x15 grid) |
| Maze seed | 42 |
| Generations | 60 (main), 20 (displaced) |
| Agents per generation | 20 |
| Training episodes | 100 |
| Max steps per episode | 400 |
| Learning rate | 0.2 |
| Discount factor | 0.95 |
| Epsilon decay | 0.5 -> 0.05 |
| Morphic emission strength | 0.5 |
| Elite fraction (genetic) | 30% |
| Mutation std | 0.05 |

## License

| Component | License |
|---|---|
| Simulation code (Python) | [Apache License 2.0](LICENSE-CODE) |
| Paper and documentation | [CC BY 4.0](LICENSE) |
| Data outputs (results, figures) | [CC BY 4.0](LICENSE) |

## Citation

```bibtex
@article{butters2026morphic,
  title   = {Collective Habit as Infrastructure: A Computational Model
             of Morphic Resonance via Multi-Agent Reinforcement Learning},
  author  = {{BUTTERS Research Group}},
  year    = {2026},
  url     = {https://github.com/web3guru888/morphic-resonance-sim}
}
```

## Related Work

- [GraphPalace](https://github.com/web3guru888/GraphPalace) --- Stigmergic memory palace engine (same design principle in embedding space)
- [MemPalace Scientific Analysis](https://github.com/web3guru888/mempalace-scientific-analysis) --- Critical analysis of spatial memory architectures for AI
