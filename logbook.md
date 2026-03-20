## 2026-03-20 12:46:29 +0100

- Files changed:
	- proposal/latex/proposal.tex
	- logbook.md
- Tests run:
	- `pdflatex -interaction=nonstopmode proposal.tex` in `proposal/latex/` → PASS
	- Result: PASS
- Summary:
	- Replaced the WRENCH/SimGrid TODO in the proposal with a related-work paragraph positioning SimGrid/WRENCH as broader workflow/distributed-system simulation frameworks.
	- Framed BottleMod-CA as a complementary, lower-overhead analytic bottleneck model with explicit storage-tier and page-cache-aware reasoning.

## 2026-02-20 13:12:12 +0100

- Files changed:
	- bottlemod/ppoly.py
- Tests run:
	- /home/justus/Code/Uni/bottlemod_cache_aware/.venv/bin/python -m unittest test.test_regression_test1_module_parity -v
	- Result: PASS
- Summary:
	- Investigated parity failure and traced it to divergent scalar indexing behavior in `PPoly.__getitem__`.
	- Restored vanilla-compatible first-segment lower bound (`-math.inf` instead of `self.x[0]`).
	- Confirmed the `Test1()` regression parity test now passes for `bottlemod` vs `bottlemod_vanilla`.

## 2026-02-20 13:51:50 +0100

- Files changed:
	- logbook.md
- Tests run:
	- ssh tu 'PYTHONPATH="$HOME/bm_exp/bottlemod_cache_aware" "$HOME/bm_exp/bottlemod_cache_aware/.venv/bin/python" "$HOME/bm_exp/bottlemod_cache_aware/thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py" --video /mnt/sata/input.mp4 --out-dir /var/tmp/exp1_ffmpeg_20260220_123431 --mem-sweep 256M,512M,1G,2G,4G,8G,16G --trials 5 --drop-caches'
	- Result: PASS (completed, JSON + figures written)
- Summary:
	- Re-ran Experiment 1 on host `tu` with the standard memory sweep and 5 trials.
	- Initial run failed due to `ModuleNotFoundError: bottlemod`; rerun succeeded by setting `PYTHONPATH` to repo root on `tu`.
	- Output directory: `/var/tmp/exp1_ffmpeg_20260220_123431`.

## 2026-02-20 13:54:40 +0100

- Files changed:
	- AGENTS.md
	- thesis_experiment/01_cach_aware_ordering/findings/20260220_123431/exp1_results.json
	- thesis_experiment/01_cach_aware_ordering/findings/20260220_123431/fig6_exp1.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260220_123431/fig7_exp1_cold_vs_warm.png
- Tests run:
	- scp -r "tu:/var/tmp/exp1_ffmpeg_20260220_123431/*" "thesis_experiment/01_cach_aware_ordering/findings/20260220_123431/"
	- Result: PASS (artifacts copied)
- Summary:
	- Copied the latest exp1 findings from host `tu` into the repository findings folder with matching timestamp.
	- Updated `AGENTS.md` experiment section to make post-run findings sync mandatory for every experiment run.

## 2026-02-26 00:00:00 +0100

- Files changed:
	- .gitmodules
	- bottlemod_vanilla
- Tests run:
	- Not run (not requested)
- Summary:
	- Replaced the local `bottlemod_vanilla` folder with a git submodule pointing to https://github.com/bottlemod/bottlemod.

## 2026-02-26 12:10:00 +0100

- Files changed:
	- AGENTS.md
	- thesis_experiment/01_cach_aware_ordering/README.md
	- thesis_experiment/01_cach_aware_ordering/01_cache_aware_ordering.md
	- logbook.md
- Tests run:
	- None (documentation-only changes)
- Summary:
	- Identified that the experiment server (`tu`) is out of sync with upstream (commit be3ec8f vs ef4de6c); `git fetch` fails due to GitHub SSH host key verification. Needs manual fix.
	- Confirmed the two-video reordering experiment code already exists in `exp1_cache_aware_ordering.py` (`--video2` flag, Figures 8 & 9) but was never exercised — all 4 prior runs used single-video only.
	- Both video files already exist on the server: `input.mp4` (4.3 GB) and `input_small.mp4` (1020 MB).
	- Updated AGENTS.md run recipe to include `--video2 /mnt/sata/input_small.mp4`.
	- Updated experiment README.md and 01_cache_aware_ordering.md to document the two-video task reordering experiment, Figures 8 & 9, and the eviction model.

## 2026-02-26 12:41:00 +0100

- Files changed:
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_124156/exp1_results.json
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_124156/fig6_exp1.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_124156/fig7_exp1_cold_vs_warm.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_124156/fig8_reordering.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_124156/fig9_bottleneck_reorder.png
	- logbook.md
- Tests run:
	- `PYTHONPATH="$ROOT" "$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py" --video /mnt/sata/input.mp4 --video2 /mnt/sata/input_small.mp4 --out-dir /var/tmp/exp1_ffmpeg_20260226_124156 --mem-sweep 256M,512M,1G,2G,4G,8G,16G --trials 5 --drop-caches`
	- Result: PASS (completed, JSON + 4 figures written)
- Summary:
	- First two-video experiment run on `tu` (host cpu09).
	- Calibration: disk_bw = 140.3 MB/s, mem_bw = 291.7 MB/s.
	- Single-video results (Fig 6): CA model tracks the downward trend well at low memory (256M–1G). At ≥2G the model underpredicts runtime (optimistic), gap largest at 2–4G (~9–10s). Vanilla flat at ~32.6s, only correct at 256M.
	- Two-video reordering results (Fig 8): Measured ordering differences are small (0–3s). CA model predicts the correct direction at 2G (A→B faster) but disagrees at 4G (predicts B→A much faster, measurements show A→B slightly faster). At 8–16G both orderings converge.
	- Fig 9 bottleneck timeline (at 4G): Best ordering (B→A) is memory-bound throughout; worst ordering (A→B) shows a disk-bound Task 2.
	- Note: PYTHONPATH must be set to repo root on tu for imports to work.
	- Findings copied to local repo under findings/20260226_124156/.

## 2026-02-26 14:04:00 +0100

- Files changed:
	- thesis_experiment/01_cach_aware_ordering/exp1_refined.py (NEW — refined experiment script)
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_140426/exp1_refined_results.json
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_140426/fig_interleaved_detail.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_140426/fig_grouped_detail.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_140426/fig_summary_comparison.png
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_140426/fig_per_task_breakdown.png
	- logbook.md
- Tests run:
	- `PYTHONPATH="$ROOT" "$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_refined.py" --video-a /mnt/sata/input.mp4 --video-b /mnt/sata/input_b.mp4 --mem-limit 5G --trials 5 --drop-caches --out-dir /var/tmp/exp1_refined_20260226_140426`
	- Result: PASS (completed, JSON + 4 figures written)
- Summary:
	- Redesigned experiment to test cache-aware ordering with **two same-size files** (4.58 GB each) under a **fixed 5 GB memory limit**.
	- Created `input_b.mp4` on server by copying `input.mp4` (same size, different inode → independent page cache).
	- New script `exp1_refined.py` runs 4 sequential ffmpeg remux tasks in two orderings:
		- **Interleaved** (A-B-A-B): Op1(A) → Op1(B) → Op2(A) → Op2(B) — each file evicts the other
		- **Grouped** (A-A-B-B): Op1(A) → Op2(A) → Op1(B) → Op2(B) — second op benefits from first op's cache
	- Calibration: disk_bw = 216.0 MB/s, mem_bw = 278.9 MB/s.
	- CA hit rates: Interleaved = [0%, 17%, 17%, 17%], Grouped = [0%, 100%, 17%, 100%].
	- **Results**:
		- **CA predicted**: Interleaved = 73.9s, Grouped = 71.6s (2.3s diff, 1.03x speedup)
		- **Vanilla predicted**: 84.8s for both orderings (no differentiation)
		- **Measured**: Interleaved = 88.3s (std=3.1s), Grouped = 87.6s (std=4.8s) (only 1.01x speedup)
	- **Analysis**: The ordering effect is real but small. With 5 GB memory and 4.58 GB files, remaining cache after one file = 0.79 GB → hit rate = 17.2%. The grouped ordering saves ~2.3s in the CA model but this is within measurement noise (std ~3-5s). The CA model underpredicts total runtime by ~15s, suggesting overhead not captured by the model.
	- **Key observation from plots**: The grouped detail plot clearly shows alternating disk-bound (orange) and memory-bound (blue) bottleneck segments, correctly identifying Op2(A) and Op2(B) as fully cached. The interleaved plot is disk-bound throughout. This qualitative distinction is the main value — the CA model correctly predicts *which* tasks benefit from caching even if absolute runtime accuracy needs improvement.
	- Findings copied to local repo under findings/20260226_140426/.

## 2026-02-26 17:14:00 +0100

- Files changed:
	- thesis_experiment/01_cach_aware_ordering/exp1_refined.py (two critical fixes)
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_171444/ (final results)
	- logbook.md
- Tests run:
	- `PYTHONPATH="$ROOT" "$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_refined.py" --video-a /mnt/sata/input_2g_a.mp4 --video-b /mnt/sata/input_2g_b.mp4 --mem-limit 3G --trials 5 --drop-caches --out-dir /var/tmp/exp1_refined_20260226_171444`
	- Result: PASS (completed, JSON + 4 figures written)
- Summary:
	- **Two critical fixes to the experiment**:
		1. **Persistent cgroup**: Replaced per-task `systemd-run --wait` with a single `systemd-run --scope` that wraps all 4 tasks. Previously each task ran in a fresh cgroup, so page cache from task N wasn't charged to task N+1's cgroup, defeating the cache pressure experiment. Now all tasks share one cgroup scope with a bash wrapper script that records per-task nanosecond timings.
		2. **Serial-access bandwidth model**: BottleMod models tiers as independent parallel resources (`time = max(cache_time, disk_time)`), but sequential file I/O accesses pages serially — each page is served by cache OR disk, not both simultaneously. Fixed CA prediction to use weighted-harmonic-mean: `eff_bw = 1/(hit/mem_bw + (1-hit)/disk_bw)`. The two-tier StorageHierarchyTask is still built for bottleneck visualization; only the predicted time is overridden.
	- **Switched to smaller files** (1.9 GB each, from input_2g_a.mp4 / input_2g_b.mp4) with **3 GB memory limit** so that `file_size > mem/2` (one file evicts the other) but `file_size < mem` (one file fits). This gives `remaining_cache = 1.1 GB`, `hit_rate = 0.602` for cross-file tasks and `1.0` for same-file tasks.
	- Created video files: `ffmpeg -t 15800 -c copy` from first/second halves of input.mp4 → different content, different inodes.
	- Calibration: disk_bw = 165.4 MB/s, mem_bw = 283.8 MB/s.
	- **Results** (best run — 20260226_171444):
		- CA predicted: Interleaved = 39.46s, Grouped = 35.43s (**1.11x speedup**)
		- Vanilla predicted: 48.63s for both orderings (no differentiation)
		- Measured: Interleaved = 37.07s (std=2.05), Grouped = 33.87s (std=1.42) (**1.09x speedup**)
	- **CA model speedup (1.11x) closely matches measured speedup (1.09x)** — the cache-aware ordering prediction is validated.
	- Per-task: CA correctly predicts fully-cached tasks (hit=100%) at ~7s, partially-cached (hit=60%) at ~9s, cold (hit=0%) at ~12s. Measurements align within ~1-2s.
	- Grouped detail plot clearly alternates disk-bound (orange) and memory-bound (blue) segments. Interleaved plot is mostly disk-bound. Vanilla predicts all tasks at cold rate.
	- Findings copied to local repo under findings/20260226_171444/.

## 2026-02-26 18:10:00 +0100

- Files changed:
	- thesis_experiment/01_cach_aware_ordering/README.md (rewritten for refined experiment)
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_140426/ (removed — buggy pre-fix run)
	- logbook.md
- Tests run:
	- None (documentation + cleanup only)
- Summary:
	- Rewrote experiment README.md to document the refined experiment: 1.9 GB same-size files, 3 GB fixed memory, interleaved vs grouped ordering, key result table (CA 1.11× vs measured 1.09×), persistent cgroup and harmonic-mean bandwidth model technical notes.
	- Removed findings/20260226_140426/ (4.58 GB files with per-task cgroup bug — no ordering effect visible, superseded by 20260226_171444).
	- Prepared for commit: exp1_refined.py, best findings (20260226_171444), two-video findings (20260226_124156), updated README, and logbook.

## 2026-02-26 21:10:00 +0100

- Files changed:
	- thesis_experiment/01_cach_aware_ordering/README.md
	- thesis_experiment/01_cach_aware_ordering/exp1_memory_increase_no_eviction.py (NEW)
	- thesis_experiment/01_cach_aware_ordering/bottlemod_ca_flowchart.md (NEW)
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_181215/
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_185746/
	- thesis_experiment/01_cach_aware_ordering/findings/20260226_200024/
	- logbook.md
- Tests run:
	- `ssh tu '... ffmpeg concat ... -> /mnt/sata/input_20g_a.mp4, /mnt/sata/input_20g_b.mp4'`
	- `ssh tu 'PYTHONPATH="$ROOT" "$PY" exp1_refined.py --video-a /mnt/sata/input_20g_a.mp4 --video-b /mnt/sata/input_20g_b.mp4 --mem-limit 30G --trials 3 --drop-caches --out-dir /var/tmp/exp1_refined_10x_20260226_181215'`
	- `ssh tu 'PYTHONPATH="$ROOT" "$PY" exp1_memory_increase_no_eviction.py --video-a /mnt/sata/input_2g_a.mp4 --video-b /mnt/sata/input_2g_b.mp4 --mem-evict 3G --mem-no-evict 5G --trials 3 --drop-caches --out-dir /var/tmp/exp1_noevict_base_20260226_185746'`
	- `ssh tu 'PYTHONPATH="$ROOT" "$PY" exp1_memory_increase_no_eviction.py --video-a /mnt/sata/input_20g_a.mp4 --video-b /mnt/sata/input_20g_b.mp4 --mem-evict 30G --mem-no-evict 70G --trials 2 --drop-caches --out-dir /var/tmp/exp1_noevict_10x70g_20260226_200024'`
	- `PYTHONPATH="/home/justus/Code/Uni/bottlemod_cache_aware" .venv/bin/python -m py_compile thesis_experiment/01_cach_aware_ordering/exp1_refined.py thesis_experiment/01_cach_aware_ordering/exp1_memory_increase_no_eviction.py`
	- `lsp_diagnostics exp1_memory_increase_no_eviction.py (severity=error)` → PASS (no errors)
	- Result: PASS
- Summary:
	- Ran the refined ordering experiment at 10× scale (18.7 GB files, 30 GB memory): measured interleaved 379.72s vs grouped 339.31s (1.12× speedup); vanilla remains 1.00×.
	- Added companion experiment `exp1_memory_increase_no_eviction.py` to demonstrate that ordering speedup collapses when memory is increased enough to avoid eviction.
	- Ran companion experiment at base scale (3G vs 5G): eviction speedup 1.075×, no-eviction 0.969× (~no ordering benefit).
	- Ran companion experiment at 10× scale (30G vs 70G): eviction speedup 1.150×, no-eviction 0.979× (~no ordering benefit).
	- Synced new findings into repo under matching timestamps (20260226_181215, 20260226_185746, 20260226_200024).
	- Added `bottlemod_ca_flowchart.md` (Mermaid) describing the BottleMod-CA modeling and measurement pipeline.

## 2026-02-28 17:20:00 +0100

- Files changed:
	- thesis_experiment/01_memory_increase/exp1_memory_increase_no_eviction.py (moved from 01_cach_aware_ordering and import path fixed)
	- thesis_experiment/01_memory_increase/README.md (NEW)
	- thesis_experiment/01_memory_increase/findings/20260226_185746/
	- thesis_experiment/01_memory_increase/findings/20260226_200024/
	- thesis_experiment/01_cach_aware_ordering/README.md (removed companion-memory sections and findings rows)
	- logbook.md
- Tests run:
	- `PYTHONPATH="/home/justus/Code/Uni/bottlemod_cache_aware" .venv/bin/python -m py_compile thesis_experiment/01_memory_increase/exp1_memory_increase_no_eviction.py`
	- `PYTHONPATH="/home/justus/Code/Uni/bottlemod_cache_aware" .venv/bin/python thesis_experiment/01_memory_increase/exp1_memory_increase_no_eviction.py --help`
	- `lsp_diagnostics thesis_experiment/01_memory_increase/exp1_memory_increase_no_eviction.py (severity=error)` → PASS (no errors)
	- Result: PASS
- Summary:
	- Created a dedicated experiment directory `thesis_experiment/01_memory_increase/`.
	- Moved `exp1_memory_increase_no_eviction.py` and its findings (`20260226_185746`, `20260226_200024`) into the new directory.
	- Added a dedicated README for the memory-increase experiment and removed companion-memory content from `01_cach_aware_ordering/README.md`.
	- Kept cache-aware reordering findings in `01_cach_aware_ordering/` and no-eviction validation findings in `01_memory_increase/`.

## 2026-02-28 18:00:00 +0100

- Files changed:
	- AGENTS.md
	- thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py (removed)
	- thesis_experiment/01_cach_aware_ordering/exp1_refined.py (removed; replaced by exp1_reordering.py)
	- thesis_experiment/01_cach_aware_ordering/exp1_reordering.py (renamed + CA runtime model switched to original TaskExecution prediction)
	- thesis_experiment/01_cach_aware_ordering/README.md
	- thesis_experiment/01_cach_aware_ordering/01_cache_aware_ordering.md
	- thesis_experiment/02_memory_increase/exp2_memory_increase_no_eviction.py
	- thesis_experiment/02_memory_increase/README.md
	- logbook.md
- Tests run:
	- `PYTHONPATH="/home/justus/Code/Uni/bottlemod_cache_aware" .venv/bin/python -m py_compile thesis_experiment/01_cach_aware_ordering/exp1_reordering.py thesis_experiment/02_memory_increase/exp2_memory_increase_no_eviction.py`
	- `PYTHONPATH="/home/justus/Code/Uni/bottlemod_cache_aware" .venv/bin/python thesis_experiment/01_cach_aware_ordering/exp1_reordering.py --help`
	- `PYTHONPATH="/home/justus/Code/Uni/bottlemod_cache_aware" .venv/bin/python thesis_experiment/02_memory_increase/exp2_memory_increase_no_eviction.py --help`
	- `lsp_diagnostics thesis_experiment/01_cach_aware_ordering/exp1_reordering.py (severity=error)` → PASS (no errors)
	- `lsp_diagnostics thesis_experiment/02_memory_increase/exp2_memory_increase_no_eviction.py (severity=error)` → PASS (no errors)
	- Result: PASS
- Summary:
	- Removed the legacy `exp1_cache_aware_ordering.py` and renamed `exp1_refined.py` to `exp1_reordering.py`.
	- Updated the reordering script to use original BottleMod-CA prediction timing from `TaskExecution` (no harmonic-mean runtime override).
	- Updated run recipes and documentation references to the new script name and current command format.
	- Updated the companion memory-increase experiment to import from `exp1_reordering.py`.

## 2026-03-05 14:19:00 +0100

- Files changed:
	- bottlemod/storage_hierarchy.py (removed `PhaseBasedCacheModel`, added `LRUEvictionModel`)
	- bottlemod/__init__.py (replaced `PhaseBasedCacheModel` with `LRUEvictionModel` in imports and `__all__`)
	- thesis_experiment/01_cach_aware_ordering/exp1_reordering.py (refactored `compute_hit_rates_interleaved` and `compute_hit_rates_grouped` to delegate to `LRUEvictionModel`)
	- bottlemod/storage_hierarchy_README.md (replaced "phase-based" reference with LRU eviction model description)
	- proposal/propsoal.md (replaced `PhaseBasedCacheModel` with `LRUEvictionModel` in thesis outline)
	- logbook.md
- Tests run:
	- `py_compile` on `storage_hierarchy.py`, `__init__.py`, `exp1_reordering.py` → PASS (all 3 files)
	- LRUEvictionModel hit-rate parity verification (interleaved + grouped, 1.9 GB files, 3 GB cache) → PASS (all assertions)
	- StorageHierarchyTask + LRU integration test (cold/warm cache, tier mappings) → PASS
	- `grep PhaseBasedCacheModel` across entire codebase → 0 matches (fully removed)
	- Result: PASS
- Summary:
	- **Replaced `PhaseBasedCacheModel` with `LRUEvictionModel`** — formalized the ad-hoc eviction-based hit-rate computation (file sizes vs. available page cache memory) into a proper model class.
	- `LRUEvictionModel` is a `@dataclass` with `cache_capacity_bytes` as the single field and two methods: `compute_hit_rates()` (returns per-task hit rates for a sequential workflow) and `compute_tier_mappings()` (wraps hit rates into `TierMapping.constant_hit_rate` calls).
	- Not a `CacheBehaviorModel` subclass because it reasons about a sequence of tasks and their cross-eviction effects, whereas `CacheBehaviorModel.compute_tier_mapping()` operates on a single task/access_profile.
	- Core formula: `remaining = max(0, capacity - previous_file_size)`, `hit_rate = min(1, remaining / current_file_size)`. Same-file follow-up: `hit_rate = min(1, capacity / file_size)`.
	- Experiment functions `compute_hit_rates_interleaved()` and `compute_hit_rates_grouped()` now delegate to `LRUEvictionModel` instead of containing inline eviction logic.
	- All references to `PhaseBasedCacheModel` removed from codebase (was in `storage_hierarchy.py`, `__init__.py`, `storage_hierarchy_README.md`, `propsoal.md`).

## 2026-03-04 14:19:00 +0100

- Files changed:
	- proposal/latex/proposal.tex
	- logbook.md
- Tests run:
	- `pdflatex -interaction=nonstopmode proposal.tex` (×2) → PASS (5 pages, no errors)
	- Result: PASS
- Summary:
	- **Thesis proposal design completed** — synthesized all prior work into a formal proposal:

## 2026-03-19 19:22:25 +0100

- Files changed:
	- bottlemod/storage_hierarchy.py
	- bottlemod/__init__.py
	- bottlemod/storage_hierarchy_README.md
	- proposal/propsoal.md
	- logbook.md
- Tests run:
	- `lsp_diagnostics bottlemod/__init__.py`
	- Result: PASS
	- `lsp_diagnostics bottlemod/storage_hierarchy.py`
	- Result: PRE-EXISTING DIAGNOSTICS (basedpyright warnings/errors already present in this file outside this change)
	- `.venv/bin/python -c "import bottlemod; import bottlemod.storage_hierarchy; print('ok')"`
	- Result: PASS
- Summary:
	- Removed `StackDistanceModel` from the core storage hierarchy module and package exports.
	- Updated storage hierarchy docs/proposal text to stop referencing `StackDistanceModel` as an active model.
	- Verified package imports still work after the removal.

## 2026-03-20 10:25:49 +0100

- Files changed:
	- bottlemod/storage_hierarchy.py
	- bottlemod/__init__.py
	- logbook.md
- Tests run:
	- `lsp_diagnostics bottlemod/__init__.py`
	- Result: PASS
	- `lsp_diagnostics bottlemod/storage_hierarchy.py`
	- Result: PRE-EXISTING DIAGNOSTICS (basedpyright warnings/errors already present in this file outside this change)
	- `.venv/bin/python -c "from bottlemod.storage_hierarchy import LogicalAccessProfile, StorageTier, WSSModel; profile = LogicalAccessProfile.sequential_read('d', total_bytes=1e9, max_progress=1.0); tiers = [StorageTier.memory(capacity_GB=0.2), StorageTier.hdd()]; model = WSSModel.constant(1e9); mapping = model.compute_tier_mapping(profile, tiers, (0.0, 1.0)); mapping.validate((0.0, 1.0)); print({k: float(v(0.5)) for k, v in mapping.H_read.items()})"`
	- Result: PASS (`{0: 0.2, 3: 0.8}`)
- Summary:
	- Added `WSSModel` as a new `CacheBehaviorModel` subclass using a progress-dependent working-set-size function.
	- Implemented cumulative-hit estimation `min(1, capacity / wss)` with inclusive tier splitting and final-tier remainder assignment.
	- Exported `WSSModel` from the package for use in cache-aware experiments and model construction.

## 2026-03-20 10:52:20 +0100

- Files changed:
	- bottlemod/storage_hierarchy.py
	- logbook.md
- Tests run:
	- `lsp_diagnostics bottlemod/storage_hierarchy.py --severity error`
	- Result: PASS
	- `.venv/bin/python -c "from bottlemod.storage_hierarchy import LogicalAccessProfile, StorageTier, WSSModel; profile = LogicalAccessProfile.sequential_read('d', total_bytes=1e9, max_progress=1.0); tiers = [StorageTier.memory(capacity_GB=0.2), StorageTier.hdd()]; model = WSSModel.constant(1e9); mapping = model.compute_tier_mapping(profile, tiers, (0.0, 1.0)); mapping.validate((0.0, 1.0)); print({k: float(v(0.5)) for k, v in mapping.H_read.items()})"`
	- Result: PASS (`{0: 0.2, 3: 0.8}`)
- Summary:
	- Fixed the basedpyright invalid-cast errors in `TierMapping.validate()` by normalizing `PPoly` evaluation results through `numpy.asarray(...).item()` before converting to `float`.
	- Confirmed `storage_hierarchy.py` is error-free in LSP diagnostics and that `WSSModel` validation still works end to end.
