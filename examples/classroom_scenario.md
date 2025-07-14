# Example: High School Classroom Group Formation

## Scenario Setup

30 students need to form project groups. We have data on:
- Academic performance (GPA)
- Social connections (friendship survey)
- Interests (sports, music, gaming, etc.)
- Personality traits (introvert/extrovert)

Initial state: Students sitting randomly, teacher announces "form groups of 4-6 for semester project"

## Theory Predictions

### CFT (Consensus-Fracture Theory)
**Parameters**: τ = 0.6 (moderate threshold)

**Prediction Process**:
1. Calculate all pairwise affinities based on weighted sum of factors
2. Find maximal cliques where all pairs meet threshold
3. Result: 6 groups of 5 students each

**Specific Prediction**:
- Groups form quickly (5 minutes)
- Very stable once formed
- Clear boundaries - no one "between" groups
- One student might be isolated if no compatible group

**Visualization**:
```
Group 1: [A,B,C,D,E] - all athletes
Group 2: [F,G,H,I,J] - all honor students  
Group 3: [K,L,M,N,O] - all gamers
etc...
```

### GFT (Gradient Field Theory)
**Parameters**: k=0.1 (attraction strength), σ=2.0 (interaction range)

**Prediction Process**:
1. Students occupy positions in multi-dimensional trait space
2. Experience attractive forces toward similar others
3. Drift toward equilibrium positions
4. Groups emerge as clusters in trait space

**Specific Prediction**:
- Gradual movement (15 minutes)
- Some students between groups
- Groups have fuzzy boundaries
- Final configuration: 5-7 groups of varying sizes (3-7 students)

**Visualization**:
```
     [A,B,C]
    /       \
  [D,E]    [F,G,H]  ← F is 70% with this group, 30% with [D,E]
```

### QST (Quantum Social Theory)  
**Parameters**: ℏ_social = 0.3 (social uncertainty), measurement_strength = 0.8

**Prediction Process**:
1. Each student in superposition of multiple potential groups
2. Observation (teacher checking) collapses possibilities
3. Entangled friends likely collapse to same group
4. Some randomness in final configuration

**Specific Prediction**:
- Uncertain until teacher walks by
- Different outcomes if teacher starts from different sides of room
- Best friends always together (entangled)
- 5-6 groups, sizes vary by measurement order

**Visualization**:
```
Before observation: Student A = 0.4|Group1⟩ + 0.6|Group2⟩
After observation: Student A → Group2 (collapsed)
```

### ICT (Information Cascade Theory)
**Parameters**: bandwidth = 4 (max meaningful connections), info_decay = 0.1/minute

**Prediction Process**:
1. Students share information about project ideas
2. Groups form around compelling ideas
3. Size limited by communication overhead
4. Cascades as students hear about "cool" groups

**Specific Prediction**:
- Forms in waves (idea spreads)
- First group with good idea gets 6-7 students (too big)
- Splits due to communication overload
- Final: 7 groups of 4-5 (optimal for information flow)

**Visualization**:
```
t=0: Random
t=5: "Let's do robotics!" → cascade
t=10: Robotics group too big, splits
t=15: Stable configuration
```

### TST (Thermodynamic Social Theory)
**Parameters**: T = 2.0 (moderate social temperature), J = 1.0 (interaction strength)

**Prediction Process**:
1. Students as particles with interaction energy
2. System minimizes free energy F = E - TS
3. Temperature determines exploration vs exploitation
4. Boltzmann distribution of configurations

**Specific Prediction**:
- Probabilistic outcome
- Most likely: 6 groups of 5
- Group size distribution follows power law
- Some chance of unusual configurations

**Visualization**:
```
P(config) ∝ exp(-E(config)/kT)
Most probable: balanced groups
Small chance: one huge group + many tiny
```

## Measurable Differences

| Observation | CFT | GFT | QST | ICT | TST |
|------------|-----|-----|-----|-----|-----|
| Formation time | 5 min | 15 min | Instant at measurement | 15 min (waves) | 10 min |
| Group boundaries | Sharp | Fuzzy | Probabilistic | Clear but dynamic | Statistical |
| Size variance | Very low | Medium | Depends on measurement | Low (optimal size) | Power law |
| Stability | Very high | Medium | Measurement-dependent | Cascade-vulnerable | Temperature-dependent |
| Predictability | Deterministic | Deterministic | Probabilistic | Path-dependent | Statistical |

## How to Test

1. **Video record** actual classroom group formation
2. **Pre-survey** student traits and relationships
3. **Track** formation dynamics minute-by-minute
4. **Post-survey** final group satisfaction
5. **Compare** predictions to reality

## Expected Reality

Likely a combination:
- **CFT aspects**: Friends definitely group together
- **GFT aspects**: Some gradual negotiation
- **QST aspects**: Teacher influence matters
- **ICT aspects**: Good ideas attract members
- **TST aspects**: Some randomness in final configuration

## This Reveals

1. Different theories capture different aspects of reality
2. Some theories better for different phases (formation vs stability)
3. Hybrid models might be necessary
4. Context determines which theory dominates

This concrete example shows how the same scenario produces testably different predictions across theories, enabling empirical validation of which mathematical framework best captures human group dynamics.