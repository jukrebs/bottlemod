# Research Prompt: Extending BottleMod for I/O Bottleneck Analysis

## Research Objective

Develop a theoretically sound extension to the BottleMod framework that models storage hierarchy effects (disk, cache/memory) on workflow execution performance. The extension must:
1. Maintain consistency with BottleMod's existing formalism and principles
2. Enable accurate prediction of I/O-related bottlenecks
3. Be empirically validatable through controlled experiments

## Background

### Current BottleMod Framework
BottleMod models workflow execution through:
- **Progress metric** p: arbitrary but consistent measure of task completion
- **Requirement functions**: 
  - Data requirements R_D,k(n_D,k) → p (storable inputs)
  - Resource requirements R_R,ℓ(p) (non-storable, e.g., CPU cycles)
- **Input functions**:
  - Data availability I_D,k(t) (monotonically increasing)
  - Resource rates I_R,ℓ(t) (instantaneous capacity)
- **Progress calculation** P(t): derived by combining data limitations (upper bound) and resource limitations (rate constraints)
- **Output functions** O_m(p): enable process composition in DAGs

### The Gap
Current BottleMod treats data as either "available" or "not available" with one-time loading assumptions. It does not model:
- Multi-tiered storage hierarchy (disk → memory/cache → CPU)
- Variable data access speeds depending on storage tier
- Cache behavior (hits, misses, evictions, thrashing)
- Access pattern effects (sequential vs. random)
- Working set size vs. cache size relationships
- Cold-to-warm cache transitions

This limits applicability to I/O-intensive workflows where storage hierarchy is the primary bottleneck.

## Research Tasks

### Task 1: Theoretical Model Development

**Objective**: Design a formal extension to BottleMod that captures storage hierarchy effects.

**Requirements**:
- Preserve BottleMod's separation of concerns (process requirements vs. execution environment)
- Maintain composability across multiple processes
- Enable clear bottleneck identification (disk vs. memory vs. CPU)
- Keep the model tractable for analysis and implementation

**Key Design Questions**:
1. How should storage tiers be represented within the requirement/input function paradigm?
   - As separate resource types with different rates?
   - As a single resource with variable effective rate?
   - Through data requirement splitting mechanisms?

2. How should cache behavior be modeled?
   - Explicit hit rate functions h(p)?
   - Implicit through working set size and cache size parameters?
   - State-based (cold/warm cache states)?

3. How should access patterns be captured?
   - As parameters to requirement functions?
   - As modifiers to effective I/O rates?
   - Through spatial/temporal locality metrics?

4. How does the extended model integrate with existing progress calculation (Section 3.2)?
   - Additional resource constraints in the iterative algorithm?
   - Modified data availability functions?
   - New constraint types?

**Deliverables**:
- Formal mathematical specification of extended requirement/input functions
- Modified progress calculation algorithm P(t) incorporating I/O constraints
- Proof or argument that the extension reduces to standard BottleMod when I/O is not limiting
- Worked examples showing how the model captures key I/O scenarios

### Task 2: Model Parameterization

**Objective**: Identify the minimal set of parameters needed to instantiate the model for practical use.

**Questions to Address**:
- What must be specified about the process (developer knowledge)?
  - Data size, working set size, access pattern, temporal reuse?
- What must be specified about the execution environment (system/admin knowledge)?
  - Cache sizes, disk bandwidth, memory bandwidth, eviction policies?
- Can parameters be derived from measurements or profiling?
- What are reasonable defaults or heuristics for common scenarios?

**Deliverables**:
- Parameter taxonomy with clear definitions
- Guidelines for parameter specification or measurement
- Sensitivity analysis: which parameters most affect predictions?

### Task 3: Experimental Validation Plan

**Objective**: Design a experimental protocol to validate model predictions against real execution.

**Deliverables**:
- Design of experiments to show the effectiveness of the bottelmod extension. Experimental environment and measurement protocol
    - In these two workload categories: Sequential Access, Random Access
- Design of appropriate scenarios in which the experiment shows the gained accuracy of the model extension compared to the original bottelmod paper

## Success Criteria

1. **Theoretical Soundness**: Model is mathematically consistent, preserves BottleMod principles, and handles edge cases correctly
2. **Predictive Accuracy**: Model predictions within 20% error for execution time on benchmark workloads
3. **Bottleneck Identification**: Model correctly identifies limiting resource (disk/memory/CPU) in >90% of test cases
4. **Actionable Insights**: Model correctly predicts which resource upgrade provides speedup in intervention experiments
5. **Generalizability**: Model applies to diverse workload types (sequential, random, mixed)



---

**Research Goal Summary**: Develop and validate a theoretically grounded, experimentally verified extension to BottleMod that enables accurate modeling and prediction of I/O bottlenecks in data-intensive workflows, maintaining consistency with the existing framework while providing actionable insights for resource allocation optimization.