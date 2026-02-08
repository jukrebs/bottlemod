from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast


def _run(
    cmd: list[str],
    *,
    cwd: Optional[str] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _maybe_run(cmd: list[str]) -> str:
    try:
        return _run(cmd).stdout.strip()
    except Exception:
        return ""


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass
class FioJob:
    name: str
    output_path: Path
    args: list[str]


def _fio_write_testfile(fio: str, filename: Path, size: str) -> None:
    # Use direct=1 for the write to avoid polluting page cache.
    _ = _run(
        [
            fio,
            f"--name=prep_write",
            f"--filename={filename}",
            "--rw=write",
            "--bs=1M",
            "--iodepth=32",
            "--ioengine=libaio",
            "--direct=1",
            f"--size={size}",
            "--numjobs=1",
            "--group_reporting",
        ]
    )


def _drop_caches() -> None:
    # Requires root. We intentionally run via sudo to avoid embedding privileged ops.
    _ = _run(["sync"])
    _ = _run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])


def _build_systemd_run_cmd(
    cmd: list[str],
    *,
    run_flags: list[str],
    properties: list[str],
    use_sudo: bool,
) -> list[str]:
    argv: list[str] = ["systemd-run"]

    # Ensure we block until completion and the unit is collected.
    if "--wait" not in run_flags:
        argv.append("--wait")
    if "--collect" not in run_flags:
        argv.append("--collect")

    argv.extend(run_flags)
    argv.extend([f"--property={prop}" for prop in properties])
    argv.extend(["--", *cmd])

    if use_sudo:
        return ["sudo", *argv]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fio buffered-IO experiments")
    _ = parser.add_argument("--fio", default="fio", help="fio binary (default: fio)")
    _ = parser.add_argument(
        "--out-dir",
        default="/var/tmp/bottlemod_exp",
        help="output directory for raw fio JSON (default: /var/tmp/bottlemod_exp)",
    )
    _ = parser.add_argument(
        "--filename",
        default=None,
        help="path to testfile (default: <out-dir>/testfile.bin)",
    )
    _ = parser.add_argument("--size", default="10G", help="test file size (fio size string)")
    _ = parser.add_argument("--trials", type=int, default=5, help="number of trials")
    _ = parser.add_argument(
        "--drop-caches",
        action="store_true",
        help="use sudo drop_caches between cold runs (stronger, requires sudo)",
    )
    _ = parser.add_argument("--seq-bs", default="1M")
    _ = parser.add_argument("--seq-iodepth", default="32")
    _ = parser.add_argument("--seq-numjobs", default="1")
    _ = parser.add_argument("--rand-bs", default="4k")
    _ = parser.add_argument("--rand-iodepth", default="64")
    _ = parser.add_argument("--rand-numjobs", default="1")

    _ = parser.add_argument(
        "--systemd-run",
        action="store_true",
        help="run fio jobs under systemd-run for cgroup v2 controls",
    )
    _ = parser.add_argument(
        "--systemd-sudo",
        action="store_true",
        help="prefix systemd-run with sudo (required for some IO throttling setups)",
    )
    _ = parser.add_argument(
        "--systemd-run-flag",
        action="append",
        default=[],
        help="one systemd-run argv entry (repeatable), e.g. '--user' or '--slice=bench.slice'",
    )
    _ = parser.add_argument(
        "--systemd-property",
        action="append",
        default=[],
        help="one systemd property (repeatable), e.g. 'MemoryMax=4G' or 'IOReadBandwidthMax=/dev/nvme0n1 200M'",
    )

    args = parser.parse_args()

    fio = cast(str, args.fio)
    out_dir_s = cast(str, args.out_dir)
    filename_s = cast(Optional[str], args.filename)
    size = cast(str, args.size)
    trials = cast(int, args.trials)
    drop_caches = cast(bool, args.drop_caches)
    seq_bs = cast(str, args.seq_bs)
    seq_iodepth = cast(str, args.seq_iodepth)
    seq_numjobs = cast(str, args.seq_numjobs)
    rand_bs = cast(str, args.rand_bs)
    rand_iodepth = cast(str, args.rand_iodepth)
    rand_numjobs = cast(str, args.rand_numjobs)

    use_systemd_run = cast(bool, args.systemd_run)
    systemd_sudo = cast(bool, args.systemd_sudo)
    systemd_run_flags = cast(list[str], args.systemd_run_flag)
    systemd_properties = cast(list[str], args.systemd_property)

    out_dir = Path(out_dir_s)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(filename_s) if filename_s else out_dir / "testfile.bin"

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_unix_s": time.time(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "uname": _maybe_run(["uname", "-a"]),
        },
        "fio": {
            "binary": fio,
            "version": _maybe_run([fio, "--version"]),
        },
        "filesystem": {
            "df_T": _maybe_run(["df", "-T", str(out_dir)]),
        },
        "params": {
            "out_dir": str(out_dir),
            "filename": str(filename),
            "size": size,
            "trials": trials,
            "drop_caches": bool(drop_caches),
            "seq": {
                "bs": seq_bs,
                "iodepth": seq_iodepth,
                "numjobs": seq_numjobs,
            },
            "rand": {
                "bs": rand_bs,
                "iodepth": rand_iodepth,
                "numjobs": rand_numjobs,
            },
            "systemd_run": {
                "enabled": bool(use_systemd_run),
                "sudo": bool(systemd_sudo),
                "run_flags": systemd_run_flags,
                "properties": systemd_properties,
            },
        },
        "runs": [],
    }

    if not filename.exists():
        _fio_write_testfile(fio, filename, size)

    jobs: list[FioJob] = []
    for trial in range(1, trials + 1):
        # Cold sequential read: invalidate=1 + optionally drop caches.
        if drop_caches:
            _drop_caches()
        seq_cold_out = out_dir / f"seqread_buffered_cold_trial{trial}.json"
        jobs.append(
            FioJob(
                name=f"seqread_buffered_cold_trial{trial}",
                output_path=seq_cold_out,
                args=[
                    fio,
                    f"--name=seqread_buffered_cold",
                    f"--filename={filename}",
                    "--rw=read",
                    f"--bs={seq_bs}",
                    f"--iodepth={seq_iodepth}",
                    "--ioengine=libaio",
                    "--direct=0",
                    "--invalidate=1",
                    f"--size={size}",
                    f"--numjobs={seq_numjobs}",
                    "--group_reporting",
                    "--output-format=json",
                    f"--output={seq_cold_out}",
                ],
            )
        )

        # Warm sequential read: run immediately after cold, do NOT invalidate.
        seq_warm_out = out_dir / f"seqread_buffered_warm_trial{trial}.json"
        jobs.append(
            FioJob(
                name=f"seqread_buffered_warm_trial{trial}",
                output_path=seq_warm_out,
                args=[
                    fio,
                    f"--name=seqread_buffered_warm",
                    f"--filename={filename}",
                    "--rw=read",
                    f"--bs={seq_bs}",
                    f"--iodepth={seq_iodepth}",
                    "--ioengine=libaio",
                    "--direct=0",
                    "--invalidate=0",
                    f"--size={size}",
                    f"--numjobs={seq_numjobs}",
                    "--group_reporting",
                    "--output-format=json",
                    f"--output={seq_warm_out}",
                ],
            )
        )

        # Random 4KiB reads: buffered, cold.
        if rand_numjobs != "0":
            if drop_caches:
                _drop_caches()
            rand_out = out_dir / f"randread_buffered_cold_trial{trial}.json"
            jobs.append(
                FioJob(
                    name=f"randread_buffered_cold_trial{trial}",
                    output_path=rand_out,
                    args=[
                        fio,
                        f"--name=randread_buffered_cold",
                        f"--filename={filename}",
                        "--rw=randread",
                        f"--bs={rand_bs}",
                        f"--iodepth={rand_iodepth}",
                        "--ioengine=libaio",
                        "--direct=0",
                        "--invalidate=1",
                        f"--size={size}",
                        f"--numjobs={rand_numjobs}",
                        "--group_reporting",
                        "--output-format=json",
                        f"--output={rand_out}",
                    ],
                )
            )

    failures: list[dict[str, object]] = []
    for job in jobs:
        if job.output_path.exists():
            job.output_path.unlink()

        invoked_cmd = (
            _build_systemd_run_cmd(
                job.args,
                run_flags=systemd_run_flags,
                properties=systemd_properties,
                use_sudo=systemd_sudo,
            )
            if use_systemd_run
            else job.args
        )

        result = _run(invoked_cmd, check=False)
        if result.returncode != 0:
            failures.append({"name": job.name, "returncode": result.returncode})
        cast(list[object], manifest["runs"]).append(
            {
                "name": job.name,
                "output": str(job.output_path),
                "cmd": job.args,
                "invoked_cmd": invoked_cmd,
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }
        )

    _write_json(out_dir / "manifest.json", manifest)

    if failures:
        raise SystemExit(f"fio_runner: {len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
    main()
