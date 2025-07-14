# Consensus-Fracture Theory (CFT)

A novel mathematical framework for modeling spontaneous group formation and fracturing based on pairwise affinities.

## Overview

Consensus-Fracture Theory provides a rigorous mathematical approach to understanding how groups form, stabilize, and fragment in complex systems. The framework introduces:

- **Bipolar affinity measures** capturing both attraction and repulsion between agents
- **Consensus-Fracture Equilibria (CFE)** for stable group configurations
- **Hierarchical extensions (HCFE)** for multi-level coalition formation
- **Multiple equilibrium refinement criteria** for practical applications

## Core Concepts

### Agents and Affinities
- **Agents**: Individual entities in the system
- **Affinity**: Pairwise measure α_ij ∈ [-1,1] where positive values indicate attraction and negative values indicate repulsion
- **Consensus Threshold**: τ, the minimum affinity required for group membership

### Consensus-Fracture Equilibrium
A stable partition of agents where:
- Within each group, all pairwise affinities meet the threshold: α_ij ≥ τ
- No further mergers are possible without violating the threshold condition

### Hierarchical Extension
Groups can form coalitions based on meta-affinities computed as average pairwise affinities between group members.

## Applications

CFT is applicable across multiple domains:
- **Biology**: Immune cell clustering, microbial community dynamics
- **Social Sciences**: Political coalition formation, social network communities
- **Economics**: Corporate alliances, market coalitions
- **AI/ML**: Clustering algorithms, multi-agent systems

## Repository Structure

```
├── paper.tex           # Main theoretical paper
├── README.md          # This file
├── ROADMAP.md         # Development roadmap
├── PROBLEMS.md        # Known issues and limitations
├── ADRLog.md          # Architectural decision records
├── CONTRIBUTING.md    # Development guidelines
└── src/               # Implementation code (future)
```

## Getting Started

### Prerequisites
- LaTeX distribution for building the paper
- Python 3.8+ (for future implementations)
- Git for version control

### Building the Paper
```bash
pdflatex paper.tex
```

## Current Status

The project is in the **ideation and theoretical development phase**. The core mathematical framework is outlined in `paper.tex`, with ongoing work on:
- Computational algorithms for finding CFEs
- Empirical validation methods
- Software implementations

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and how to contribute to the project.

## License

[To be determined - pending discussion]

## Contact

[Author information to be added]

