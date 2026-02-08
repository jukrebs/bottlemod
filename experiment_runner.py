from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

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


@dataclass
class ExperimentResult:
    name: str
    analytic_runtime_s: float
    upstream_runtime_s: float
    sh_runtime_s: float
    upstream_error_pct: float
    sh_error_pct: float
    sh_bottleneck: str


def _data_available_input(max_progress: float, time_range: Tuple[float, float]) -> Func:
    return Func([time_range[0], time_range[1]], [[0, max_progress]])


def _run_task(
    cpu_funcs: List[PPoly],
    cpu_inputs: List[PPoly],
    max_progress: float,
    time_range: Tuple[float, float],
) -> float:
    data_func = Func([0, max_progress], [[1, 0]])
    data_input = _data_available_input(max_progress, time_range)
    task = Task(cpu_funcs, [data_func])
    execution = TaskExecution(task, cpu_inputs, [data_input])
    progress_func, _ = execution.get_result()
    return float(progress_func.x[-1])


def _filter_zero_requirements(
    requirements: List[PPoly], inputs: List[PPoly]
) -> Tuple[List[PPoly], List[PPoly]]:
    filtered_requirements = []
    filtered_inputs = []
    for requirement, input_func in zip(requirements, inputs):
        if all(cc == 0 for cc in requirement.c.flatten()):
            continue
        filtered_requirements.append(requirement)
        filtered_inputs.append(input_func)
    return filtered_requirements, filtered_inputs


def _error_pct(predicted: float, analytic: float) -> float:
    if analytic == 0:
        return 0.0
    return (predicted - analytic) / analytic * 100.0


def _sequential_scan() -> ExperimentResult:
    file_size = 10 * 1024 * 1024 * 1024
    time_range = (0.0, 1000.0)
    disk_bw = 500 * 1024 * 1024
    max_progress = file_size

    analytic = file_size / disk_bw

    upstream_cpu_funcs = [PPoly([0, max_progress], [[1]])]
    upstream_cpu_inputs = [PPoly([time_range[0], time_range[1]], [[disk_bw]])]
    upstream_time = _run_task(upstream_cpu_funcs, upstream_cpu_inputs, max_progress, time_range)

    access_profile = LogicalAccessProfile.sequential_read(
        name="input_file",
        total_bytes=file_size,
        max_progress=max_progress,
    )
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=25.0, time_range=time_range),
        StorageTier.sata_ssd(name="SSD", bandwidth_MBps=500.0, time_range=time_range),
    ]
    tier_mapping = TierMapping.all_from_tier(
        dataset_name="input_file",
        tier_index=2,
        progress_range=(0, max_progress),
    )
    storage_reqs, storage_inputs = derive_tier_resources(access_profile, tier_mapping, tiers)
    storage_reqs, storage_inputs = _filter_zero_requirements(storage_reqs, storage_inputs)
    sh_time = _run_task(storage_reqs, storage_inputs, max_progress, time_range)

    return ExperimentResult(
        name="Sequential scan (no reuse)",
        analytic_runtime_s=analytic,
        upstream_runtime_s=upstream_time,
        sh_runtime_s=sh_time,
        upstream_error_pct=_error_pct(upstream_time, analytic),
        sh_error_pct=_error_pct(sh_time, analytic),
        sh_bottleneck="disk_bw",
    )


def _two_pass_cold_warm() -> ExperimentResult:
    file_size = 4 * 1024 * 1024 * 1024
    time_range = (0.0, 100.0)
    disk_bw = 500 * 1024 * 1024
    mem_bw = 25 * 1024 * 1024 * 1024
    max_progress = 2 * file_size
    warmup_progress = file_size

    analytic = file_size / disk_bw + file_size / mem_bw

    upstream_cpu_funcs = [PPoly([0, max_progress], [[1]])]
    upstream_cpu_inputs = [PPoly([time_range[0], time_range[1]], [[disk_bw]])]
    upstream_time = _run_task(upstream_cpu_funcs, upstream_cpu_inputs, max_progress, time_range)

    access_profile = LogicalAccessProfile.sequential_read(
        name="dataset",
        total_bytes=max_progress,
        max_progress=max_progress,
    )
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=25.0, capacity_GB=16.0, time_range=time_range),
        StorageTier.sata_ssd(name="SSD", bandwidth_MBps=500.0, time_range=time_range),
    ]
    tier_mapping = TierMapping.cold_then_warm(
        dataset_name="dataset",
        cold_tier=2,
        warm_tier=0,
        warmup_progress=warmup_progress,
        progress_range=(0, max_progress),
        warm_hit_rate=0.999,
    )
    storage_reqs, storage_inputs = derive_tier_resources(access_profile, tier_mapping, tiers)
    storage_reqs, storage_inputs = _filter_zero_requirements(storage_reqs, storage_inputs)
    sh_time = _run_task(storage_reqs, storage_inputs, max_progress, time_range)

    return ExperimentResult(
        name="Two-pass cold/warm cache",
        analytic_runtime_s=analytic,
        upstream_runtime_s=upstream_time,
        sh_runtime_s=sh_time,
        upstream_error_pct=_error_pct(upstream_time, analytic),
        sh_error_pct=_error_pct(sh_time, analytic),
        sh_bottleneck="disk_bw -> mem_bw",
    )


def run_experiments() -> List[ExperimentResult]:
     return [_sequential_scan(), _two_pass_cold_warm()]


def main() -> None:
    results = run_experiments()
    table = []
    for result in results:
        table.append(
            {
                "experiment": result.name,
                "analytic_runtime_s": result.analytic_runtime_s,
                "upstream_runtime_s": result.upstream_runtime_s,
                "sh_runtime_s": result.sh_runtime_s,
                "upstream_error_pct": result.upstream_error_pct,
                "sh_error_pct": result.sh_error_pct,
                "sh_bottleneck": result.sh_bottleneck,
            }
        )
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
