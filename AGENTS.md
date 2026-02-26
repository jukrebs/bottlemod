# Agent Instructions

## Virtual Environment Setup
- Always use the virtual environment when working on this project
- If the virtual environment is not present, create it by running: `./setup_bottlemod_env.sh`
- Note: Running `setup_bottlemod_env.sh` recreates `.venv`; only rerun when a rebuild is intended.

## Scope Guardrails
- Prefer editing core code in `bottlemod/`.
- Treat `bottlemod_vanilla/` as a reference/baseline unless explicitly requested.

## Dependency/Install Fallback
- If `requirements.txt` is missing, document the install issue and proceed with minimal required packages only.

## Reproducibility and Output Hygiene
- Do not modify historical experiment outputs under `thesis_experiment/`; write new results to a new timestamped folder.
- Do not edit generated artifacts such as `__pycache__/`.

## Logging Work
- After completing work, always log what you have done in `logbook.md`
- Include a timestamp and summary in bulletpoints.
- Each entry must include: timestamp, files changed, tests run, and a short bullet summary.

## Running experiments

### Mandatory host and execution target
- Run thesis experiments on the SSH environment `tu` (compute node currently used: `cpu09`).
- Do not use a local laptop/desktop run as a thesis result source.
- Keep host identity in run metadata (`platform.node()` is already captured by experiment scripts).

### Hardware and storage baseline (tu)
- CPU: AMD EPYC 7282 (16 cores)
- RAM: 125 GiB
- Primary experiment disk: `/dev/sda` mounted at `/mnt/sata` (ext4, ~880 GB)
- Reference dataset path for exp1: `/mnt/sata/input.mp4` (single large sequential-read file)

### Reproducibility limits and constraints on tu
- Use cgroup v2 memory limits to control effective page cache size (e.g. `256M ... 16G`).
- Use `--drop-caches` between runs when measuring cold behavior.
- Keep input data on `/mnt/sata` for all runs (do not mix disks across trials).
- Treat measured storage speed as environment-specific:
	- Calibrate per run and store `disk_bw_bytes_s` / `mem_bw_bytes_s` in result JSON.
	- Do not hardcode prior bandwidth values as universal constants.
- Avoid concurrent heavy I/O on `tu` during measurements to reduce contention noise.
- Write new outputs to a new timestamped folder (e.g. `/var/tmp/exp1_ffmpeg_YYYYmmdd_HHMMSS`).

### Standard exp1 run recipe on tu
- Project env:
	- If missing, create `.venv` via `./setup_bottlemod_env.sh`.
	- Use `.venv/bin/python` explicitly.
- Run:
	- `ROOT="$HOME/bm_exp/bottlemod_cache_aware"`
	- `PY="$ROOT/.venv/bin/python"`
	- `"$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py" --video /mnt/sata/input.mp4 --out-dir "/var/tmp/exp1_ffmpeg_$(date +%Y%m%d_%H%M%S)" --mem-sweep 256M,512M,1G,2G,4G,8G,16G --trials 5 --drop-caches`

### Post-run findings sync (mandatory)
- After every experiment run on `tu`, copy the run artifacts from `/var/tmp/exp1_ffmpeg_YYYYmmdd_HHMMSS` into this repository under `thesis_experiment/01_cach_aware_ordering/findings/YYYYmmdd_HHMMSS/`.
- Keep the original timestamp in the destination folder name.
- Recommended copy command from local host:
	- `TS=YYYYmmdd_HHMMSS`
	- `mkdir -p thesis_experiment/01_cach_aware_ordering/findings/$TS`
	- `scp -r "tu:/var/tmp/exp1_ffmpeg_$TS/*" "thesis_experiment/01_cach_aware_ordering/findings/$TS/"`

### Minimal run metadata to document for every experiment
- Hostname, CPU model, RAM size
- Disk device + mountpoint used for input data
- cgroup memory sweep values
- Calibrated storage rates (`disk_bw_bytes_s`, `mem_bw_bytes_s`)
- ffmpeg version and command-line flags

