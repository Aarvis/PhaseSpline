from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple spline websocket servers on one machine.")
    parser.add_argument("--config", required=True, help="Base spline server config.")
    parser.add_argument("--num-servers", type=int, required=True, help="Number of spline servers to launch.")
    parser.add_argument("--start-port", type=int, default=9100, help="First websocket port.")
    parser.add_argument("--gpu-id", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value for all child servers.")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable used for child processes.")
    parser.add_argument("--stagger-seconds", type=float, default=2.0, help="Delay between launches.")
    parser.add_argument("--log-dir", type=str, default="logs/multi_serve_spline_policy", help="Directory for logs and run metadata.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Forwarded KEY=VALUE config overrides.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_servers <= 0:
        raise ValueError("--num-servers must be > 0.")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "config": str(Path(args.config).expanduser().resolve()),
        "num_servers": int(args.num_servers),
        "start_port": int(args.start_port),
        "gpu_id": args.gpu_id,
        "stagger_seconds": float(args.stagger_seconds),
        "servers": [],
    }

    processes: list[subprocess.Popen[bytes]] = []
    print(f"[multi_serve_spline_policy] run_dir={run_dir}")
    for index in range(args.num_servers):
        port = args.start_port + index
        log_path = run_dir / f"server_{index:02d}_port_{port}.log"
        cmd = [
            args.python_exe,
            "serve_spline_policy.py",
            "--config",
            args.config,
            "--port",
            str(port),
        ]
        for item in args.overrides:
            cmd.extend(["--set", item])

        metadata["servers"].append(
            {
                "index": index,
                "port": port,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )

        print(f"[multi_serve_spline_policy] server[{index}] port={port} log={log_path}")
        if args.dry_run:
            continue

        env = dict(os.environ)
        if args.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        log_handle = open(log_path, "wb")
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        time.sleep(max(0.0, args.stagger_seconds))

    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        return

    endpoints = [f"ws://127.0.0.1:{args.start_port + index}" for index in range(args.num_servers)]
    print("[multi_serve_spline_policy] endpoints:")
    for endpoint in endpoints:
        print(f"  {endpoint}")

    try:
        exit_code = 0
        for process in processes:
            code = process.wait()
            if code != 0 and exit_code == 0:
                exit_code = code
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        print("[multi_serve_spline_policy] interrupt received, terminating child processes...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
