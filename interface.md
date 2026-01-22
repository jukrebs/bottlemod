# BottleMod Interface: PPoly and Theoretical Model Mapping

This document explains how the **PPoly** (Piecewise Polynomial) interface in BottleMod translates to the theoretical model described in the paper (`paper/bottlemod.md`).

---

## Overview

BottleMod models task execution by combining:
1. **Requirement Functions** — How much resource/data is needed to reach a given progress $p$
2. **Input Functions** — How much resource/data is available over time $t$

The `PPoly` class implements piecewise polynomial functions that represent these mathematical constructs.

---

## PPoly Constructor

```python
PPoly(x: ndarray, c: ndarray)
```

- **`x`**: Breakpoints (domain boundaries) — shape `(m+1,)` for `m` segments
- **`c`**: Coefficients — shape `(k, m)` where `k` is the polynomial order

### Coefficient Interpretation

For a segment $[x_i, x_{i+1}]$, the polynomial is:
$$f(x) = c_0 \cdot x^{k-1} + c_1 \cdot x^{k-2} + \ldots + c_{k-1}$$

For **linear functions** (most common in BottleMod), `c` has shape `(2, m)`:
- `c[0][i]` = slope of segment $i$
- `c[1][i]` = intercept of segment $i$

---

## Mapping to Theoretical Model

### Resource Requirement Function $R_{R,\ell}(p)$

**Paper Definition**: $R_{R,\ell}(p)$ gives the cumulative resource needed to reach progress $p$.

**PPoly Implementation**:
```python
cpu_requirement = PPoly(
    [0, 50, 100],      # Progress breakpoints
    [[1, 2], [2, 2]]   # Coefficients: [slopes], [intercepts]
)
```

This defines:
- Segment 1: $R(p) = 1p + 2$ for $p \in [0, 50]$
- Segment 2: $R(p) = 2p + 2$ for $p \in [50, 100]$

The **derivative** $R'(p)$ gives the instantaneous resource rate per progress unit:
- $R'(p) = 1$ for $p \in [0, 50]$ — needs 1 CPU per progress unit
- $R'(p) = 2$ for $p \in [50, 100]$ — needs 2 CPU per progress unit

### Resource Input Function $I_{R,\ell}(t)$

**Paper Definition**: $I_{R,\ell}(t)$ describes the resource rate available at time $t$.

**PPoly Implementation**:
```python
cpu_input = PPoly(
    [0, 1000],    # Time breakpoints
    [[2], [2]]    # Constant rate of 2 CPU units per time
)
```

This defines:
- $I(t) = 2t + 2$ for $t \in [0, 1000]$
- Derivative $I'(t) = 2$ — constant supply of 2 CPU units per time unit

### Data Requirement Function $R_{D,k}(n_{D,k}) \mapsto p$

**Paper Definition**: Maps consumed data to achievable progress.

**PPoly Implementation**:
```python
data_dep = PPoly(
    [0, 100],     # Data amount breakpoints
    [[1], [0]]    # Linear: 1 data unit = 1 progress unit
)
```

This defines $R_D(n) = 1 \cdot n + 0$, meaning progress equals data consumed.

### Data Input Function $I_{D,k}(t)$

**Paper Definition**: Available data over time; monotonically increasing.

**PPoly Implementation**:
```python
data_input = PPoly(
    [0, 1000],     # Time breakpoints  
    [[0], [100]]   # Constant: 100 units available instantly
)
```

This defines $I_D(t) = 100$ for all $t \geq 0$ — data is immediately available.

---

## Worked Example: `example_simple.py`

### Setup

```python
# Resource requirement: CPU
cpu_requirement_1 = PPoly([0, 50, 100], [[1, 2], [2, 2]])

# Data requirement: 1:1 mapping
data_dep = PPoly([0, 100], [[1], [0]])

# Resource input: constant rate 2
cpu_input_1 = PPoly([0, 1000], [[2], [2]])

# Data input: 100 units instantly
data_input_1 = PPoly([0, 1000], [[0], [100]])
```

### Analysis

**Phase 1: Progress $p \in [0, 50]$**

| Quantity | Value |
|----------|-------|
| Resource requirement rate | $R'(p) = 1$ CPU per progress |
| Resource supply rate | $I'(t) = 2$ CPU per time |
| Progress speed | $\frac{I'(t)}{R'(p)} = \frac{2}{1} = 2$ progress/time |
| Time to complete | $\frac{50}{2} = 25$ time units |

**Phase 2: Progress $p \in [50, 100]$**

| Quantity | Value |
|----------|-------|
| Resource requirement rate | $R'(p) = 2$ CPU per progress |
| Resource supply rate | $I'(t) = 2$ CPU per time |
| Progress speed | $\frac{I'(t)}{R'(p)} = \frac{2}{2} = 1$ progress/time |
| Time to complete | $\frac{50}{1} = 50$ time units |

**Total execution time**: $25 + 50 = 75$ time units

### Physical Interpretation

The task models work that:
1. **Starts easy** (low CPU demand per progress unit)
2. **Becomes harder** (doubles CPU demand in the second half)

With constant CPU supply, the task **slows down** in the second half because more resources are needed per unit of progress.

---

## Key Paper Equations and PPoly Operations

### Progress Speed (Paper §3.2)

$$\text{Speed} = \frac{I_{R,\ell}(t)}{R'_{R,\ell}(P(t))}$$

In PPoly terms:
- `cpu_input(t, nu=1)` → $I'_{R,\ell}(t)$ (derivative of input)
- `cpu_requirement(p, nu=1)` → $R'_{R,\ell}(p)$ (derivative of requirement)

### Resource Utilization (Paper §3.3)

$$U_\ell(t) = \frac{P'(t) \cdot R'_{R,\ell}(P(t))}{I_{R,\ell}(t)}$$

A utilization of $U_\ell = 1$ indicates a **bottleneck** — the resource is fully utilized.

### Speedup Factor (Paper §3.2)

$$S_{PR,\ell}(t) = \frac{I_{R,\ell}(t)}{P'(t) \cdot R'_{R,\ell}(P(t))}$$

When $S_{PR,\ell}(t) < 1$, resource $\ell$ is limiting progress.

---

## PPoly Operations Supporting the Model

| Operation | PPoly Method | Paper Concept |
|-----------|--------------|---------------|
| Composition | `f(g)` where both are PPoly | $R_D(I_D(t))$ — progress from data |
| Derivative | `f(x, nu=1)` | $R'(p)$, $I'(t)$ — rates |
| Addition | `f + g` | Combining contributions |
| Division | `f / g` | Speed = input / requirement |
| Slicing | `f[a:b]` | Extract function over range |
| Root finding | `f.roots()` | Find when functions cross zero |

---

## Summary Table

| Paper Notation | Meaning | PPoly Role |
|----------------|---------|------------|
| $R_{R,\ell}(p)$ | Cumulative resource to reach $p$ | `TaskRequirements.add_resource(requirement_func=...)` |
| $R'_{R,\ell}(p)$ | Resource rate per progress | Derivative of requirement PPoly |
| $I_{R,\ell}(t)$ | Resource rate at time $t$ | `ExecutionEnvironment.add_resource(input_func=...)` |
| $R_{D,k}(n)$ | Data consumed → progress | `TaskRequirements.add_data(dependency_func=...)` |
| $I_{D,k}(t)$ | Available data at time $t$ | `ExecutionEnvironment.add_data(input_func=...)` |
| $P(t)$ | Progress over time | Computed by `TaskExecution` |
| $U_\ell(t)$ | Resource utilization | Computed in bottleneck analysis |
