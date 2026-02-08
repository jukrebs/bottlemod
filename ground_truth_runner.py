from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


FILE_PATH = "/tmp/bottlemod_gt_10gb"
BLOCK_SIZE = 4 * 1024 * 1024
REQUEST_SIZE = 4096
RANDOM_READS = 1_000_000


@dataclass
class GroundTruthResult:
    experiment: str
    actual_runtime_s: float
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


def _error_pct(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0
    return (predicted - actual) / actual * 100.0


def _ensure_file() -> int:
    if not os.path.exists(FILE_PATH):
        raise RuntimeError(
            f"Missing {FILE_PATH}. Create it with: dd if=/dev/zero of={FILE_PATH} bs=1M count=10240"
        )
    return os.path.getsize(FILE_PATH)


def _sequential_read_runtime(file_size: int) -> float:
    buffer = bytearray(BLOCK_SIZE)
    start = time.perf_counter()
    with open(FILE_PATH, "rb", buffering=0) as handle:
        remaining = file_size
        while remaining > 0:
            to_read = BLOCK_SIZE if remaining >= BLOCK_SIZE else remaining
            view = memoryview(buffer)[:to_read]
            read_bytes = handle.readinto(view)
            if read_bytes == 0:
                break
            remaining -= read_bytes
    end = time.perf_counter()
    return end - start


def _random_read_runtime(file_size: int) -> float:
    max_offset = file_size - REQUEST_SIZE
    rng = random.Random(0)
    start = time.perf_counter()
    fd = os.open(FILE_PATH, os.O_RDONLY)
    try:
        for _ in range(RANDOM_READS):
            offset = rng.randrange(0, max_offset // REQUEST_SIZE) * REQUEST_SIZE
            os.pread(fd, REQUEST_SIZE, offset)
    finally:
        os.close(fd)
    end = time.perf_counter()
    return end - start


def _run_ground_truth() -> Dict[str, GroundTruthResult]:
    file_size = _ensure_file()
    time_range = (0.0, 100000.0)

    pass1 = _sequential_read_runtime(file_size)
    pass2 = _sequential_read_runtime(file_size)
    disk_bw = file_size / pass1
    mem_bw = file_size / pass2

    random_time = _random_read_runtime(file_size)
    disk_iops = RANDOM_READS / random_time

    sequential_upstream = _run_task(
        [PPoly([0, file_size], [[1]])],
        [PPoly([time_range[0], time_range[1]], [[disk_bw]])],
        file_size,
        time_range,
    )
    access_profile = LogicalAccessProfile.sequential_read(
        name="input_file",
        total_bytes=file_size,
        max_progress=file_size,
    )
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=mem_bw / 1e9, time_range=time_range),
        StorageTier.sata_ssd(name="SSD", bandwidth_MBps=disk_bw / 1e6, time_range=time_range),
    ]
    tier_mapping = TierMapping.all_from_tier(
        dataset_name="input_file",
        tier_index=2,
        progress_range=(0, file_size),
    )
    storage_reqs, storage_inputs = derive_tier_resources(access_profile, tier_mapping, tiers)
    storage_reqs, storage_inputs = _filter_zero_requirements(storage_reqs, storage_inputs)
    sequential_sh = _run_task(storage_reqs, storage_inputs, file_size, time_range)

    two_pass_upstream = _run_task(
        [PPoly([0, 2 * file_size], [[1]])],
        [PPoly([time_range[0], time_range[1]], [[disk_bw]])],
        2 * file_size,
        time_range,
    )
    access_profile_two = LogicalAccessProfile.sequential_read(
        name="dataset",
        total_bytes=2 * file_size,
        max_progress=2 * file_size,
    )
    tiers_two = [
        StorageTier.memory(
            name="DRAM",
            bandwidth_GBps=mem_bw / 1e9,
            capacity_GB=16.0,
            time_range=time_range,
        ),
        StorageTier.sata_ssd(
            name="SSD",
            bandwidth_MBps=disk_bw / 1e6,
            time_range=time_range,
        ),
    ]
    tier_mapping_two = TierMapping.cold_then_warm(
        dataset_name="dataset",
        cold_tier=2,
        warm_tier=0,
        warmup_progress=file_size,
        progress_range=(0, 2 * file_size),
        warm_hit_rate=0.999,
    )
    storage_reqs_two, storage_inputs_two = derive_tier_resources(
        access_profile_two, tier_mapping_two, tiers_two
    )
    storage_reqs_two, storage_inputs_two = _filter_zero_requirements(
        storage_reqs_two, storage_inputs_two
    )
    two_pass_sh = _run_task(storage_reqs_two, storage_inputs_two, 2 * file_size, time_range)

    random_upstream = _run_task(
        [PPoly([0, RANDOM_READS * REQUEST_SIZE], [[1]])],
        [PPoly([time_range[0], time_range[1]], [[disk_bw]])],
        RANDOM_READS * REQUEST_SIZE,
        time_range,
    )
    access_profile_random = LogicalAccessProfile.random_read(
        name="random_dataset",
        total_bytes=RANDOM_READS * REQUEST_SIZE,
        max_progress=RANDOM_READS * REQUEST_SIZE,
        request_size=REQUEST_SIZE,
    )
    tiers_random = [
        StorageTier.memory(
            name="DRAM",
            bandwidth_GBps=mem_bw / 1e9,
            capacity_GB=16.0,
            time_range=time_range,
        ),
        StorageTier(
            name="SSD",
            tier_index=1,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[disk_bw]]),
            I_iops_read=PPoly([time_range[0], time_range[1]], [[disk_iops]]),
            capacity=1e15,
        ),
    ]
    cache_model = StackDistanceModel.uniform_reuse(file_size)
    tier_mapping_random = cache_model.compute_tier_mapping(
        access_profile_random, tiers_random, (0, RANDOM_READS * REQUEST_SIZE)
    )
    storage_reqs_random, storage_inputs_random = derive_tier_resources(
        access_profile_random, tier_mapping_random, tiers_random
    )
    storage_reqs_random, storage_inputs_random = _filter_zero_requirements(
        storage_reqs_random, storage_inputs_random
    )
    random_sh = _run_task(
        storage_reqs_random,
        storage_inputs_random,
        RANDOM_READS * REQUEST_SIZE,
        time_range,
    )

    return {
        "sequential": GroundTruthResult(
            experiment="Sequential scan (no reuse)",
            actual_runtime_s=pass1,
            upstream_runtime_s=sequential_upstream,
            sh_runtime_s=sequential_sh,
            upstream_error_pct=_error_pct(sequential_upstream, pass1),
            sh_error_pct=_error_pct(sequential_sh, pass1),
            sh_bottleneck="disk_bw",
        ),
        "two_pass": GroundTruthResult(
            experiment="Two-pass cold/warm cache",
            actual_runtime_s=pass1 + pass2,
            upstream_runtime_s=two_pass_upstream,
            sh_runtime_s=two_pass_sh,
            upstream_error_pct=_error_pct(two_pass_upstream, pass1 + pass2),
            sh_error_pct=_error_pct(two_pass_sh, pass1 + pass2),
            sh_bottleneck="disk_bw -> mem_bw",
        ),
        "random": GroundTruthResult(
            experiment="Random 4KiB reads (IOPS-bound)",
            actual_runtime_s=random_time,
            upstream_runtime_s=random_upstream,
            sh_runtime_s=random_sh,
            upstream_error_pct=_error_pct(random_upstream, random_time),
            sh_error_pct=_error_pct(random_sh, random_time),
            sh_bottleneck="disk_iops",
        ),
    }


def main() -> None:
    results = _run_ground_truth()
    payload = {
        key: {
            "experiment": value.experiment,
            "actual_runtime_s": value.actual_runtime_s,
            "upstream_runtime_s": value.upstream_runtime_s,
            "sh_runtime_s": value.sh_runtime_s,
            "upstream_error_pct": value.upstream_error_pct,
            "sh_error_pct": value.sh_error_pct,
            "sh_bottleneck": value.sh_bottleneck,
        }
        for key, value in results.items()
    }
    with open("experiment_ground_truth.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
