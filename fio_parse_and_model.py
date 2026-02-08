from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

from bottlemod.func import Func
from bottlemod.ppoly import PPoly
from bottlemod.task import Task, TaskExecution
from bottlemod.storage_hierarchy import (
    LogicalAccessProfile,
    StorageTier,
    TierMapping,
    StackDistanceModel,
    derive_tier_resources,
)


def _read_json_maybe_prefixed(path: Path) -> object:
    # fio can emit non-JSON lines in some failure modes; be tolerant.
    raw = path.read_text(encoding="utf-8", errors="replace")
    start = raw.find("{")
    if start > 0:
        raw = raw[start:]
    return cast(object, json.loads(raw))


def _parse_size_bytes(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    suffixes = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    for suf, mul in suffixes.items():
        if s.lower().endswith(suf):
            num = s[: -len(suf)].strip()
            return int(float(num) * mul)
    raise ValueError(f"Unsupported size string: {s!r}")


def _median(values: Iterable[float]) -> float:
    vv = list(values)
    if not vv:
        return 0.0
    return float(statistics.median(vv))


@dataclass
class AggregatedFio:
    wall_runtime_s: float
    read_bw_bytes_s: float
    read_iops: float
    read_io_bytes: int


def _aggregate_fio_jobs(payload: dict[str, object]) -> AggregatedFio:
    jobs_obj = payload.get("jobs")
    jobs = jobs_obj if isinstance(jobs_obj, list) else None
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("fio JSON missing jobs[]")

    job_runtimes_ms: list[float] = []
    bw_bytes_s: float = 0.0
    iops_total: float = 0.0
    io_bytes_total: int = 0

    for job_obj in jobs:
        if not isinstance(job_obj, dict):
            continue
        job = cast(dict[str, object], job_obj)

        read_obj = job.get("read", {})
        if not isinstance(read_obj, dict):
            continue
        read = cast(dict[str, object], read_obj)

        # Prefer read.runtime (per-job wall-clock time in ms) over job_runtime.
        # With --group_reporting, job_runtime may be summed across jobs in a group.
        # read.runtime gives the actual wall-clock time per job.
        runtime_ms = 0.0
        runtime_obj = read.get("runtime", None)
        if runtime_obj is not None and isinstance(runtime_obj, (int, float)):
            runtime_ms = float(runtime_obj)
        else:
            # Fallback to job_runtime if read.runtime is not available
            job_runtime_obj = job.get("job_runtime", 0)
            runtime_ms = float(job_runtime_obj) if isinstance(job_runtime_obj, (int, float)) else 0.0
        job_runtimes_ms.append(runtime_ms)

        bw_obj = read.get("bw_bytes", 0.0)
        bw_bytes_s += float(bw_obj) if isinstance(bw_obj, (int, float)) else 0.0

        iops_obj = read.get("iops", 0.0)
        iops_total += float(iops_obj) if isinstance(iops_obj, (int, float)) else 0.0

        io_bytes_obj = read.get("io_bytes", 0)
        io_bytes_total += int(io_bytes_obj) if isinstance(io_bytes_obj, int) else 0

    wall_s = max(job_runtimes_ms) / 1000.0 if job_runtimes_ms else 0.0
    return AggregatedFio(
        wall_runtime_s=wall_s,
        read_bw_bytes_s=bw_bytes_s,
        read_iops=iops_total,
        read_io_bytes=io_bytes_total,
    )


def _data_available_input(max_progress: float, time_range: tuple[float, float]) -> Func:
    return Func([time_range[0], time_range[1]], [[0, max_progress]])


def _run_task(
    cpu_funcs: list[PPoly],
    cpu_inputs: list[PPoly],
    max_progress: float,
    time_range: tuple[float, float],
) -> float:
    data_func = Func([0, max_progress], [[1, 0]])
    data_input = _data_available_input(max_progress, time_range)
    task = Task(cpu_funcs, [data_func])
    execution = TaskExecution(task, cpu_inputs, [data_input])
    progress_func, _ = execution.get_result()
    return float(progress_func.x[-1])


def _error_pct(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0
    return (predicted - actual) / actual * 100.0


def _filter_zero_requirements(requirements: list[PPoly], inputs: list[PPoly]) -> tuple[list[PPoly], list[PPoly]]:
    filtered_requirements: list[PPoly] = []
    filtered_inputs: list[PPoly] = []
    for requirement, input_func in zip(requirements, inputs):
        if all(cc == 0 for cc in requirement.c.flatten()):
            continue
        filtered_requirements.append(requirement)
        filtered_inputs.append(input_func)
    return filtered_requirements, filtered_inputs


def _predict_sequential(file_size: int, disk_bw: float) -> tuple[float, float]:
    max_progress = float(file_size)
    time_range = (0.0, 1e6)
    upstream = _run_task(
        [PPoly([0, max_progress], [[1]])],
        [PPoly([time_range[0], time_range[1]], [[disk_bw]])],
        max_progress,
        time_range,
    )

    access_profile = LogicalAccessProfile.sequential_read(
        name="input_file",
        total_bytes=float(file_size),
        max_progress=max_progress,
    )
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=25.0, time_range=time_range),
        StorageTier(
            name="NVMe",
            tier_index=1,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[disk_bw]]),
            I_iops_read=PPoly([time_range[0], time_range[1]], [[1e12]]),
            capacity=1e18,
        ),
    ]
    mapping = TierMapping.all_from_tier(
        dataset_name="input_file",
        tier_index=1,
        progress_range=(0, max_progress),
    )
    storage_reqs, storage_inputs = derive_tier_resources(access_profile, mapping, tiers)
    storage_reqs, storage_inputs = _filter_zero_requirements(storage_reqs, storage_inputs)
    sh = _run_task(storage_reqs, storage_inputs, max_progress, time_range)
    return upstream, sh


def _predict_two_pass(file_size: int, disk_bw: float, mem_bw: float) -> tuple[float, float]:
    max_progress = float(2 * file_size)
    warmup_progress = float(file_size)
    time_range = (0.0, 1e6)

    upstream = _run_task(
        [PPoly([0, max_progress], [[1]])],
        [PPoly([time_range[0], time_range[1]], [[disk_bw]])],
        max_progress,
        time_range,
    )

    access_profile = LogicalAccessProfile.sequential_read(
        name="dataset",
        total_bytes=max_progress,
        max_progress=max_progress,
    )
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=mem_bw / 1e9, capacity_GB=16.0, time_range=time_range),
        StorageTier(
            name="NVMe",
            tier_index=1,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[disk_bw]]),
            I_iops_read=PPoly([time_range[0], time_range[1]], [[1e12]]),
            capacity=1e18,
        ),
    ]
    mapping = TierMapping.cold_then_warm(
        dataset_name="dataset",
        cold_tier=1,
        warm_tier=0,
        warmup_progress=warmup_progress,
        progress_range=(0, max_progress),
        warm_hit_rate=0.999,
    )
    storage_reqs, storage_inputs = derive_tier_resources(access_profile, mapping, tiers)
    storage_reqs, storage_inputs = _filter_zero_requirements(storage_reqs, storage_inputs)
    sh = _run_task(storage_reqs, storage_inputs, max_progress, time_range)
    return upstream, sh


def _predict_random(
    file_size: int,
    read_io_bytes: int,
    request_size: int,
    disk_bw: float,
    disk_iops: float,
    dram_capacity_bytes: int,
) -> tuple[float, float]:
    max_progress = float(read_io_bytes)
    time_range = (0.0, 1e6)

    upstream = _run_task(
        [PPoly([0, max_progress], [[1]])],
        [PPoly([time_range[0], time_range[1]], [[disk_bw]])],
        max_progress,
        time_range,
    )

    access_profile = LogicalAccessProfile.random_read(
        name="random_dataset",
        total_bytes=float(read_io_bytes),
        max_progress=max_progress,
        request_size=float(request_size),
    )
    tiers = [
        StorageTier.memory(
            name="DRAM",
            bandwidth_GBps=25.0,
            capacity_GB=max(0.001, dram_capacity_bytes / 1e9),
            time_range=time_range,
        ),
        StorageTier(
            name="NVMe",
            tier_index=1,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[disk_bw]]),
            I_iops_read=PPoly([time_range[0], time_range[1]], [[disk_iops]]),
            capacity=1e18,
        ),
    ]

    cache_model = StackDistanceModel.streaming_no_reuse()
    mapping = cache_model.compute_tier_mapping(access_profile, tiers, (0, max_progress))
    storage_reqs, storage_inputs = derive_tier_resources(access_profile, mapping, tiers)
    storage_reqs, storage_inputs = _filter_zero_requirements(storage_reqs, storage_inputs)
    sh = _run_task(storage_reqs, storage_inputs, max_progress, time_range)
    return upstream, sh


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse fio JSON and run BottleMod predictions")
    parser.add_argument("--in-dir", required=True, help="directory with fio_runner outputs")
    parser.add_argument(
        "--out",
        default="experiment_ground_truth_fio.json",
        help="output json path (default: experiment_ground_truth_fio.json)",
    )
    parser.add_argument(
        "--dram-capacity",
        default="16G",
        help="DRAM/page-cache capacity for model (default: 16G; use a cgroup limit if applicable)",
    )
    parser.add_argument(
        "--rand-request-size",
        default="4k",
        help="random read request size (default: 4k)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json in {in_dir}")
    manifest_obj = _read_json_maybe_prefixed(manifest_path)
    if not isinstance(manifest_obj, dict):
        raise SystemExit("manifest.json is not a JSON object")
    manifest = cast(dict[str, object], manifest_obj)

    params_obj = manifest.get("params")
    if not isinstance(params_obj, dict):
        raise SystemExit("manifest.json missing params object")
    params = cast(dict[str, object], params_obj)

    filename_obj = params.get("filename")
    if not isinstance(filename_obj, str):
        raise SystemExit("manifest.json missing params.filename")
    testfile = Path(filename_obj)
    if not testfile.exists():
        raise SystemExit(f"Missing test file: {testfile}")
    file_size = testfile.stat().st_size

    dram_capacity_bytes = _parse_size_bytes(args.dram_capacity)
    request_size = _parse_size_bytes(args.rand_request_size)

    seq_cold_runs: list[AggregatedFio] = []
    seq_warm_runs: list[AggregatedFio] = []

    runs_obj = manifest.get("runs", [])
    runs = runs_obj if isinstance(runs_obj, list) else []
    for entry_obj in runs:
        if not isinstance(entry_obj, dict):
            continue
        entry = cast(dict[str, object], entry_obj)

        name = str(entry.get("name", ""))
        output_obj = entry.get("output")
        if not isinstance(output_obj, str):
            continue
        out_path = Path(output_obj)
        if not out_path.exists():
            continue
        payload_obj = _read_json_maybe_prefixed(out_path)
        if not isinstance(payload_obj, dict):
            continue
        payload = cast(dict[str, object], payload_obj)
        agg = _aggregate_fio_jobs(payload)
        if name.startswith("seqread_buffered_cold_trial"):
            seq_cold_runs.append(agg)
        elif name.startswith("seqread_buffered_warm_trial"):
            seq_warm_runs.append(agg)

    if not seq_cold_runs or not seq_warm_runs:
        raise SystemExit("Missing required fio runs (seq cold/warm).")

    disk_bw = _median(r.read_bw_bytes_s for r in seq_cold_runs)
    mem_bw = _median(r.read_bw_bytes_s for r in seq_warm_runs)

    # Actual runtimes: use median wall time, and for two-pass use median of per-trial sums.
    seq_actual = _median(r.wall_runtime_s for r in seq_cold_runs)

    # Two-pass: pair runs by order in manifest by trial index (same count expected).
    two_pass_sums: list[float] = []
    for cold, warm in zip(seq_cold_runs, seq_warm_runs):
        two_pass_sums.append(cold.wall_runtime_s + warm.wall_runtime_s)
    two_pass_actual = _median(two_pass_sums)

    upstream_seq, sh_seq = _predict_sequential(file_size, disk_bw)
    upstream_two, sh_two = _predict_two_pass(file_size, disk_bw, mem_bw)

    result = {
        "schema_version": 1,
        "meta": {
            "source": "fio",
            "in_dir": str(in_dir),
            "testfile": str(testfile),
            "file_size_bytes": file_size,
            "dram_capacity_bytes": dram_capacity_bytes,
            "derived": {
                "disk_bw_bytes_s_median": disk_bw,
                "mem_bw_bytes_s_median": mem_bw,
            },
        },
        "sequential": {
            "experiment": "Sequential scan (buffered, cold)",
            "actual_runtime_s": seq_actual,
            "upstream_runtime_s": upstream_seq,
            "sh_runtime_s": sh_seq,
            "upstream_error_pct": _error_pct(upstream_seq, seq_actual),
            "sh_error_pct": _error_pct(sh_seq, seq_actual),
            "sh_bottleneck": "disk_bw",
        },
        "two_pass": {
            "experiment": "Two-pass cold/warm (buffered)",
            "actual_runtime_s": two_pass_actual,
            "upstream_runtime_s": upstream_two,
            "sh_runtime_s": sh_two,
            "upstream_error_pct": _error_pct(upstream_two, two_pass_actual),
            "sh_error_pct": _error_pct(sh_two, two_pass_actual),
            "sh_bottleneck": "disk_bw -> mem_bw",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ = out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
