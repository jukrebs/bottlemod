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

