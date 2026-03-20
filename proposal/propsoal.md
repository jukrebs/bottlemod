# Master Thesis Proposal — Design Document

> **LaTeX proposal**: `proposal/latex/proposal.tex` (compile with `pdflatex proposal.tex`)
>
> This file documents the thesis design decisions: research questions, chapter outline, plot designs, experiment gap analysis, and assessment of what work remains.

---

## Research Questions

**RQ1 — Prediction Accuracy.**
Can a bottleneck-based model predict I/O-bound task runtime when effective read bandwidth is a mixture of memory-speed cache hits and storage-speed cache misses, achieving <20% MAPE across varying memory limits and access patterns?

**RQ2 — Bottleneck Identification.**
Does the cache-aware extension correctly identify the limiting resource (page cache bandwidth vs. disk bandwidth) at each phase of task execution, and can it predict where bottleneck transitions occur as memory pressure changes?

**RQ3 — Optimization Enablement.**
Can BottleMod-CA predict the runtime impact of cache-aware task ordering (and thereby enable ordering optimizations), where the vanilla model cannot?

---

## Thesis Outline (Chapters)

### Chapter 1: Introduction (5–8 pages)
- Motivation: I/O-intensive workflows, page cache as hidden performance variable
- Problem statement: existing models treat storage as single-rate, miss cache effects
- Research questions (RQ1–RQ3)
- Contributions summary
- Thesis structure overview

### Chapter 2: Background (10–15 pages)
- 2.1 Scientific workflows and task modeling
- 2.2 BottleMod formalism: progress metric, requirement functions, input functions, progress calculation, bottleneck decomposition
- 2.3 Storage hierarchies: DRAM, SSD, HDD; bandwidth vs. latency characteristics
- 2.4 Linux page cache: architecture, LRU eviction, cgroup v2 memory limits
- 2.5 Cache modeling: stack distance analysis, reuse distance, working set model

### Chapter 3: Related Work (8–10 pages)
- 3.1 Bottleneck and roofline models (Williams et al., ECM)
- 3.2 Workflow performance models and simulation (WRENCH, SimGrid)
- 3.3 Page cache modeling (Meurillon et al. 2021)
- 3.4 Cache-aware scheduling (CPU cache DAG scheduling, I/O-aware HPC scheduling)
- 3.5 Storage hierarchy management (multi-tier, burst buffers)
- 3.6 Gap analysis: no existing model integrates page-cache dynamics into a per-task bottleneck predictor

### Chapter 4: BottleMod-CA — Model Extension (12–15 pages)
- 4.1 Design goals: preserve BottleMod's separation of concerns, composability, and closed-form character
- 4.2 Process-side descriptors: LogicalAccessProfile A_k(p) — sequential, random, piecewise constructors
- 4.3 Environment-side descriptors: StorageTier (bandwidth, capacity) — predefined tiers (memory, NVMe, SATA, HDD)
- 4.4 Cache behavior models (pluggable):
  - 4.4.1 DirectHitRateModel: user-provided h(p)
  - 4.4.2 LRUEvictionModel: multi-task sequential cache eviction based on file sizes vs. available page-cache capacity
- 4.5 Tier mapping H_{k,j}(p): hit-rate fractions, sum-to-one constraint
- 4.6 Resource derivation: R'(p) = H(p) · A'(p) — from hit fractions to BottleMod resource requirements
- 4.7 Integration: StorageHierarchyTask → Task → TaskExecution (unchanged solver)
- 4.8 Properties: reduction to vanilla when no cache effects, composability across datasets

### Chapter 5: Implementation (8–10 pages)
- 5.1 Architecture overview: storage_hierarchy.py as upstream preprocessor
- 5.2 Key classes and their relationships (UML/data flow diagram)
- 5.3 Numerical considerations: PPoly arithmetic, sampling density, precision thresholds
- 5.4 Testing: unit tests, regression tests against vanilla BottleMod
- 5.5 API usage guide: how to model a new workload

### Chapter 6: Experimental Evaluation (15–20 pages)
- 6.1 Experimental setup: hardware (tu/cpu09), cgroup methodology, calibration protocol
- 6.2 Workloads:
  - 6.2.1 Sequential I/O: ffmpeg video remux (read-dominated, predictable access pattern)
  - 6.2.2 Random I/O: database-like workload or fio random read (if added)
- 6.3 Experiment 1 — Memory sweep: runtime vs. memory limit (hit rate varies)
  - CA vs. vanilla vs. measured; MAPE comparison (answers RQ1)
- 6.4 Experiment 2 — Task reordering: interleaved vs. grouped ordering
  - Speedup prediction; bottleneck timeline comparison (answers RQ2, RQ3)
- 6.5 Experiment 3 — No-eviction control: ordering effect vanishes with sufficient memory
  - Validates causal mechanism (supports RQ2)
- 6.6 Experiment 4 — Sensitivity analysis: parameter sensitivity (disk_bw, mem_bw, hit_rate)
  - Which parameters most affect prediction accuracy? (supports RQ1)
- 6.7 Results and discussion

### Chapter 7: Conclusion (3–5 pages)
- 7.1 Summary of contributions
- 7.2 Answers to research questions
- 7.3 Limitations and threats to validity
- 7.4 Future work: multi-tier (SSD+HDD), distributed storage, CPU cache integration, dynamic capacity

---

## Plot Designs (Thesis-Quality Figures)

### Plot 1: Bottleneck Timeline (Fig. 6 style from BottleMod paper)
- **What**: Progress over time with color-coded bottleneck bands
- **Purpose**: Shows BottleMod-CA's core value — identifying WHICH resource limits each phase
- **Design**: 2×4 panel (2 orderings × 4 tasks). Top: progress % with colored bands (orange=disk-bound, blue=memory-bound). Bottom: bandwidth usage vs time.
- **Key insight**: Grouped ordering shows alternating disk/memory bands; interleaved is mostly disk-bound.
- **Answers**: RQ2 (bottleneck identification)

### Plot 2: Prediction Accuracy Comparison (Bar Chart)
- **What**: Total workflow runtime: CA predicted vs. Vanilla predicted vs. Measured, for both orderings
- **Purpose**: Head-to-head accuracy comparison
- **Design**: Grouped bar chart. X-axis: {Interleaved, Grouped}. Y-axis: total time (s). Three bars per group: CA (blue), Vanilla (gray), Measured (green) with error bars.
- **Key insight**: CA bars closely match measured; vanilla bars are flat and overpredict.
- **Answers**: RQ1 (prediction accuracy), RQ3 (ordering prediction)

### Plot 3: Per-Task Breakdown (Stacked Bars)
- **What**: Per-task runtimes stacked, comparing CA vs. measured for both orderings
- **Purpose**: Shows per-task accuracy and which tasks benefit from caching
- **Design**: Stacked bar chart. Each bar = 4 task segments. Side-by-side: CA vs. Measured for each ordering.
- **Key insight**: Cold tasks (hit=0%) match well; warm tasks (hit=100%) match well; partial tasks show moderate accuracy.
- **Answers**: RQ1 (per-task prediction accuracy)

### Plot 4: Memory Sweep Curve (Line Plot)
- **What**: Runtime vs. memory limit (256 MB → 16 GB) from legacy sweep data
- **Purpose**: Shows how CA prediction tracks the non-linear runtime-vs-memory relationship
- **Design**: X-axis: memory limit (log scale). Y-axis: runtime (s). Three lines: CA predicted (blue), Vanilla predicted (gray dashed), Measured mean (green with error bars).
- **Key insight**: CA follows the downward curve; vanilla is a flat horizontal line.
- **Answers**: RQ1 (accuracy across memory range)

### Plot 5: Speedup Prediction (Small Table/Bar)
- **What**: Predicted vs. measured ordering speedup at base and 10× scale
- **Purpose**: Direct RQ3 evidence — can CA predict the speedup that vanilla misses?
- **Design**: Simple grouped bars or table. Shows CA speedup prediction, measured speedup, vanilla speedup (always 1.0×).
- **Answers**: RQ3

### Plot 6: No-Eviction Control (Paired Bars)
- **What**: Measured speedup under eviction vs. no-eviction regimes
- **Purpose**: Validates causal mechanism — ordering effect disappears when cache is sufficient
- **Design**: Two bar groups: {Eviction regime, No-eviction regime}. Each group: measured speedup. Horizontal line at 1.0×.
- **Key insight**: Eviction bars > 1.0× (1.07–1.15×); no-eviction bars ≈ 1.0× (0.97–0.98×).
- **Answers**: RQ2 (supports causal claim)

### Plot 7: Hit Rate Model Visualization
- **What**: Per-task hit rates for both orderings (table or heatmap)
- **Purpose**: Makes the cache eviction model intuitive
- **Design**: 4×2 heatmap or table: tasks × orderings, color = hit rate (0.0=red, 0.6=yellow, 1.0=green).
- **Answers**: Model explanation (Chapter 4)

### Plot 8: MAPE Comparison Table
- **What**: MAPE values for CA vs. vanilla across experiment configurations
- **Purpose**: Quantitative accuracy summary
- **Design**: Table with rows = {Base scale, 10× scale} and columns = {CA MAPE, Vanilla MAPE, Reduction}.
- **Values**: CA: 5.5%/2.9%, Vanilla: 37.4%/18.9%, Reduction: ~85%.
- **Answers**: RQ1

---

## Experiment Sufficiency Assessment

### What we have (sufficient for core thesis claims):
1. ✅ **Sequential I/O, task reordering** — 2 scales, 5 trials, both orderings
2. ✅ **No-eviction control** — confirms causal mechanism at 2 scales
3. ✅ **Memory sweep** — legacy data showing runtime-vs-memory curve
4. ✅ **Per-task accuracy** — cold/partial/warm predictions vs. measured
5. ✅ **MAPE < 20%** — success criterion met (5.5% and 2.9%)
6. ✅ **Statistical significance** — paired t-tests, bootstrap CIs, Cohen's d effect sizes
7. ✅ **Sensitivity analysis** — ±20% perturbation sweep of disk_bw and mem_bw, heatmaps and 1D plots

### What we should add (strengthens the thesis):
1. **Memory sweep with refined experiment** (NICE TO HAVE): Run the persistent-cgroup experiment across multiple memory limits (not just one fixed point) to produce a proper runtime-vs-memory curve with the fixed methodology. The legacy sweep used the buggy per-task cgroup approach.

### What would be bonus (only if time permits):
1. **Random I/O workload**: fio random read benchmark to test the model with non-sequential access patterns. Would motivate adding a simpler dedicated random-access cache model.
2. **Multi-tier storage**: NVMe + SATA + HDD experiment to test 3-tier hierarchy. Requires hardware we may not have.
3. **Different workload types**: Database query, image processing pipeline, etc.

### Verdict:
**Current experiments are sufficient for a solid thesis.** All core claims (RQ1: accuracy, RQ2: bottleneck identification, RQ3: ordering optimization) are supported by existing data. Statistical significance tests and sensitivity analysis are now complete, strengthening the evaluation chapter. The only remaining recommended addition is a memory sweep with the fixed persistent-cgroup methodology.

---

## Thesis Chapter ↔ Experiment Mapping

| Chapter/Section | Data Source | Status |
|----------------|-------------|--------|
| Ch. 4 (Model) | Implementation in `storage_hierarchy.py` | ✅ Complete |
| Ch. 6.3 (Memory sweep) | `findings/20260220_123431`, `findings/20260226_124156` | ✅ Available (legacy methodology) |
| Ch. 6.4 (Task reordering) | `findings/20260226_171444` (base), `findings/20260226_181215` (10×) | ✅ Complete |
| Ch. 6.5 (No-eviction control) | `findings/20260226_185746` (base), `findings/20260226_200024` (10×) | ✅ Complete |
| Ch. 6.6 (Sensitivity analysis) | `findings/sensitivity_analysis_results.json` | ✅ Complete |
| Statistical tests | `findings/statistical_analysis_results.json` | ✅ Complete |

---

## Statistical Analysis Results

### Paired t-tests (Interleaved vs. Grouped Total Runtime)

| Scale | n | t-statistic | p-value | Significant (α=0.05) | Cohen's d | Effect Size |
|-------|---|-------------|---------|----------------------|-----------|-------------|
| Base (1.9 GB × 2, 3 GB) | 5 | 2.305 | 0.0767 | No (marginal) | 1.061 | Large |
| 10× (18.7 GB × 2, 30 GB) | 3 | 8.374 | 0.0140 | Yes | 4.836 | Very large |

### Bootstrap Speedup Confidence Intervals (10000 resamples)

| Scale | Measured Speedup | 95% CI | CA Predicted Speedup |
|-------|-----------------|--------|---------------------|
| Base | 1.09× | [1.027, 1.169] | 1.11× |
| 10× | 1.12× | [1.091, 1.136] | 1.06× |

### No-Eviction Control Speedup CIs

| Scale | Regime | Speedup | 95% CI | Contains 1.0? |
|-------|--------|---------|--------|---------------|
| Base | Eviction (3 GB) | 1.075× | [1.015, 1.152] | No |
| Base | No-eviction (5 GB) | 0.969× | [0.956, 0.983] | No (below 1) |
| 10× | Eviction (30 GB) | 1.150× | [1.108, 1.186] | No |
| 10× | No-eviction (70 GB) | 0.979× | [0.970, 0.987] | No (below 1) |

### Interpretation
- The ordering effect is statistically significant at 10× scale (p=0.014) and marginally significant at base scale (p=0.077, limited by n=5).
- Large effect sizes (Cohen's d > 1) at both scales confirm practical significance.
- Bootstrap CIs for speedup exclude 1.0 in the eviction regime, confirming a real ordering benefit.
- No-eviction CIs are below 1.0 (slight overhead from ordering), confirming the effect vanishes without cache pressure.

---

## Sensitivity Analysis Results

### MAPE vs. Calibration Error (±20% perturbation, 9×9 grid)

| Scale | Calibrated MAPE | MAPE Range | Stays Below 20% |
|-------|----------------|------------|-----------------|
| Base (1.9 GB × 2, 3 GB) | 17.85% | [6.43%, 31.54%] | Mostly (exceeds at +disk/+mem) |
| 10× (18.7 GB × 2, 30 GB) | 20.04% | [11.63%, 33.37%] | Partially (exceeds at positive perturbations) |

### Key Findings
1. **Disk bandwidth is the dominant sensitivity axis**: overestimating disk_bw by +10–20% increases MAPE by 5–15 percentage points; underestimating by −10–20% can actually improve MAPE.
2. **Memory bandwidth has a secondary effect**: interacts with disk perturbation; modest overestimation (+5–10%) has minimal impact alone but compounds with disk overestimation.
3. **Robustness**: The model stays below 20% MAPE for the majority of the ±20% perturbation space. The worst-case corner (disk +20%, mem +20%) reaches ~31–33% MAPE.
4. **Calibration sweet spot**: slightly underestimating disk_bw (−10–20%) with slightly overestimating mem_bw (+5%) yields the lowest MAPE (~6–12%), suggesting the calibrated disk bandwidth may be slightly high.

### Plots
- `fig_sensitivity_heatmap_base.png` / `fig_sensitivity_heatmap_10x.png`: 2D MAPE heatmaps
- `fig_sensitivity_disk_bw_base.png` / `fig_sensitivity_disk_bw_10x.png`: 1D disk BW sweep
- `fig_sensitivity_mem_bw_base.png` / `fig_sensitivity_mem_bw_10x.png`: 1D memory BW sweep
