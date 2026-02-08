# experiment_reworked.md — Ground-truth experiments (NVMe host)

Host assumptions (from `lsblk`):

```
nvme0n1p2 mounted at /
RAM: 16GB
Storage: 1TB NVMe SSD (OS disk)
fio: installed
```

Goal: obtain **real ground truth runtimes** on this host for I/O-heavy workloads and compare:

1) **Upstream BottleMod (bandwidth-only)**
2) **BottleMod-SH (tiered BW + cache hit-rate mapping)**
3) **Measured runtime** from the host (fio)

This document is a plan plus exact commands.

---

## 0) Safety / reproducibility notes

- These experiments use the **OS disk**. Use a dedicated directory (below) and avoid running during critical workloads.
- We want **application-like behavior**, so we use **buffered I/O** (`--direct=0`) and treat the Linux page cache as the “DRAM tier”.
- For reproducibility, we run in a **controlled environment** (see Section 3): pinned CPUs + bounded memory (page cache budget) + optional I/O throttling.
- For cache-state control, we rely on fio invalidation knobs for “cold” vs “warm” runs. Dropping caches via `/proc/sys/vm/drop_caches` is optional and requires sudo.
- Always record:
  - kernel version (`uname -a`)
  - fio version (`fio --version`)
  - filesystem for the test directory (`df -T <dir>`)

---

## 1) Directory + test file

### 1.1 Create an experiment directory on `/`

```bash
sudo mkdir -p /var/tmp/bottlemod_exp
sudo chown "$USER":"$USER" /var/tmp/bottlemod_exp
```

### 1.2 Create a 10GiB test file (on NVMe)

Using fio (avoids dd quirks):

```bash
fio --name=prep_write --filename=/var/tmp/bottlemod_exp/testfile.bin \
    --rw=write --bs=1M --iodepth=32 --ioengine=libaio \
    --direct=1 --size=10G --numjobs=1 --group_reporting
```

Verify:

```bash
ls -lh /var/tmp/bottlemod_exp/testfile.bin
```

---

## 2) Ground-truth benchmarks (fio)

All commands below should be run multiple times (e.g., 5 trials) and report **median runtime**.

### 2.1 Sequential scan (buffered, cold page cache)

Purpose: measure effective throughput when the application reads a large file via the page cache.

```bash
fio --name=seqread_direct --filename=/var/tmp/bottlemod_exp/testfile.bin \
    --rw=read --bs=1M --iodepth=32 --ioengine=libaio \
    --direct=0 --invalidate=1 --size=10G --numjobs=1 --group_reporting \
    --output-format=json --output=/var/tmp/bottlemod_exp/seqread_buffered_cold.json
```

Extract metrics from JSON:
- runtime (ms): `jobs[].job_runtime` (or `jobs[].read.runtime` for read-only)
- bandwidth (bytes/s): `jobs[].read.bw_bytes`

### 2.2 Two-pass cold → warm (page cache effect)

Purpose: show BottleMod-SH can represent cache warming (hit-rate transition), while upstream cannot.

We want:
- **Pass 1 (cold):** ensure page cache not already warm.
- **Pass 2 (warm):** same file, immediately, should hit page cache (file is 10GiB < 16GiB RAM).

#### Option A (recommended): fio cache invalidation knobs (no sudo)

Pass 1 (invalidate page cache before run):

```bash
fio --name=seqread_buffered_cold --filename=/var/tmp/bottlemod_exp/testfile.bin \
    --rw=read --bs=1M --iodepth=32 --ioengine=libaio \
    --direct=0 --invalidate=1 --size=10G --numjobs=1 --group_reporting \
    --output-format=json --output=/var/tmp/bottlemod_exp/seqread_buffered_cold.json
```

Pass 2 (do NOT invalidate; should be warm):

```bash
fio --name=seqread_buffered_warm --filename=/var/tmp/bottlemod_exp/testfile.bin \
    --rw=read --bs=1M --iodepth=32 --ioengine=libaio \
    --direct=0 --invalidate=0 --size=10G --numjobs=1 --group_reporting \
    --output-format=json --output=/var/tmp/bottlemod_exp/seqread_buffered_warm.json
```

#### Option B (stronger, requires sudo): drop caches explicitly

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

Then run pass 1, then pass 2 without dropping caches.

---

## 3) Containerized runs (optional, for stability + controlled cache size)

Why containers help:
- consistent environment
- ability to constrain memory (simulate smaller page cache)

Example: run sequential read with memory limit (on cgroup v2, page cache is accounted per cgroup on modern distros):

```bash
docker run --rm --memory=4g --cpuset-cpus=0-3 \
  -v /var/tmp/bottlemod_exp:/data \
  ghcr.io/axboe/fio \
  fio --name=seqread_buffered --filename=/data/testfile.bin \
      --rw=read --bs=1M --iodepth=32 --ioengine=libaio \
      --direct=0 --invalidate=1 --size=10G --numjobs=1 --group_reporting
```

Notes:
- If you use container fio, keep host fio runs too; treat container runs as “controlled-variance” runs.

### 3.1 Alternative: systemd-run (no containers)

If you use systemd, you can run fio in a transient unit with bounded memory and pinned CPUs:

```bash
systemd-run --user --scope \
  -p CPUAffinity=0-3 \
  -p MemoryMax=4G \
  fio --name=seqread_buffered_cold --filename=/var/tmp/bottlemod_exp/testfile.bin \
      --rw=read --bs=1M --direct=0 --invalidate=1 --size=10G --numjobs=1 --group_reporting
```

Optional (advanced): add I/O throttling (e.g., io.max / IOReadBandwidthMax) if the host is noisy.

#### Throttled (recommended for reproducibility)

On many distros, the `io` controller is **not delegated** to `--user` units. In that case, I/O throttling requires **system scope** (sudo).

Example (read throttled + bounded memory + pinned CPUs):

```bash
python fio_runner.py \
  --out-dir /var/tmp/bottlemod_exp \
  --trials 5 \
  --systemd-run --systemd-sudo \
  --systemd-property "CPUAffinity=0-3" \
  --systemd-property "MemoryMax=4G" \
  --systemd-property "MemorySwapMax=0" \
  --systemd-property "IOReadBandwidthMax=/dev/nvme0n1 200M" \
  --rand-numjobs 0
```

Notes:
- systemd property syntax uses a **space** between device path and limit. Keep the whole property as one string:
  - `IOReadBandwidthMax=/dev/nvme0n1 200M`
- `fio_runner.py` automatically adds `systemd-run --wait --collect`.
- Pass `--rand-numjobs 0` to skip the random read experiment (we focus on sequential workloads for this evaluation).

### 3.2 Reproducibility options (recommended order)

For buffered I/O ground truth that is repeatable and comparable across hosts, choose one of these setups:

1) **systemd-run + cgroup v2 (recommended)**
   - Controls: memory (includes page cache), CPU pinning, and I/O throttling.
   - Best when you can run directly on a Linux host with systemd.

2) **Docker + cgroup v2**
   - Same controllers as (1), but with a shareable environment.
   - Best when you want a portable setup across machines.

3) **VM (KVM/QEMU)**
   - Strongest isolation, but adds overhead and complicates “page cache = DRAM tier” (guest + host caches).
   - Best when you need kernel-level isolation.

---

## 4) Mapping fio ground truth → model inputs

For each experiment, from fio JSON (`--output-format=json`):

- **Read BW (bytes/s)** = sum of `jobs[].read.bw_bytes`
- **Runtime (s)** = max of `jobs[].job_runtime` / 1000

Notes:
- fio JSON uses `job_runtime` (milliseconds). There is no `job_runtime_ms` field.
- For read-only jobs, `jobs[].read.runtime` may be used instead of `job_runtime`.

Cache warm experiment:
- Use cold pass BW as “disk tier BW”
- Use warm pass BW as “memory tier BW” proxy (effective cache-served throughput)

---

## 5) Modeling steps (what to compute)

### 5.1 Upstream BottleMod (baseline)

For each workload:
- represent the I/O requirement as a single requirement function `R(p)` (bytes per progress)
- represent the environment input as a single bandwidth function `I(t)` (bytes/s)

This model cannot represent:
- tiered cache hits

### 5.2 BottleMod-SH

For each workload:
- Define `LogicalAccessProfile`:
   - sequential scan: `A_read(p)` only
- Define tiers:
   - DRAM tier with BW from warm read
   - NVMe tier with BW from cold read
- Define mapping `H(p)`:
   - sequential scan: all from NVMe
   - cold→warm: piecewise hit rate switching after first pass

Expected:
- two-pass cold→warm: predicted bottleneck shift disk→memory

---

## 6) Evaluation

For each experiment:

| Workload | fio runtime (median) | Upstream predicted | SH predicted | Upstream error | SH error | SH bottleneck |
|---|---:|---:|---:|---:|---:|---|

Use error %:

`(predicted - actual) / actual * 100`

---

## 7) Deliverables

- `experiment_reworked.md` (this plan)
- fio JSON outputs in `/var/tmp/bottlemod_exp/*.json`
- a script to parse fio JSON and feed both models
- plots:
  - runtime bars: actual vs upstream vs SH
  - absolute error bars: upstream vs SH

---

## 8) Next implementation steps (after plan approval)

1) Implement `fio_runner.py` to run fio jobs, collect JSON
2) Implement `fio_parse_and_model.py` to:
   - parse BW/runtime
   - run both models with measured inputs
   - emit `results.json`
3) Extend plotting script to include fio ground truth

## 9) Implemented automation (repo)

This repo includes scripts to run and parse these experiments:

- `python fio_runner.py --out-dir /var/tmp/bottlemod_exp --trials 5`
- `python fio_parse_and_model.py --in-dir /var/tmp/bottlemod_exp --out experiment_ground_truth_fio.json`
- `python experiment_plots.py` (will prefer fio ground truth if present)

Convenience shell scripts (see `scripts/`):

- `bash scripts/run_fio_throttled_smoke.sh`
- `bash scripts/run_fio_throttled_2g_cache4g.sh`
- `bash scripts/run_fio_throttled_8g_cache16g.sh`
- `bash scripts/run_fio_unthrottled.sh`
- `OUT_DIR=/var/tmp/bottlemod_exp_throttled_2g_cache4g bash scripts/plot_fio_results.sh`
