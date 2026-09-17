"""NCCL communication benchmark: AllReduce, AllGather, ReduceScatter.

Uses nccl-tests binary if available, otherwise falls back to PyTorch.
"""

import os
import subprocess
import json
import re


def run_nccl_test(op: str, min_size: str = "1M", max_size: str = "256M",
                   gpus: int = 2, output_file: str = None):
    """Запускает nccl-tests для операции."""
    binary = f"./nccl-tests/build/{op}_perf"

    if not os.path.exists(binary):
        print(f"WARNING: {binary} not found. Compile nccl-tests first.")
        print("Run: git clone https://github.com/NVIDIA/nccl-tests.git")
        print("     cd nccl-tests && make MPI=0 CUDA_HOME=/usr/local/cuda")
        return None

    cmd = [binary, "-b", min_size, "-e", max_size, "-f", "2", "-g", str(gpus)]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # Парсим вывод
    # Формат: size count type redop root time algbw busbw #wrong ...
    lines = output.strip().split("\n")
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            try:
                size_bytes = int(parts[0])
                busbw = float(parts[7])
                data.append({
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / 1e6,
                    "busbw_gbps": busbw,
                })
            except (ValueError, IndexError):
                continue

    result_dict = {
        "operation": op,
        "gpus": gpus,
        "data": data,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

    return result_dict


def main():
    os.makedirs("results", exist_ok=True)

    # AllReduce (DDP gradient sync)
    ar = run_nccl_test("all_reduce", output_file="results/nccl_all_reduce.json")

    # AllGather (ZeRO parameter collection)
    ag = run_nccl_test("all_gather", output_file="results/nccl_all_gather.json")

    # ReduceScatter (ZeRO gradient sharding)
    rs = run_nccl_test("reduce_scatter", output_file="results/nccl_reduce_scatter.json")

    # Печатаем summary
    print("\n=== NCCL Benchmark Summary ===")
    for name, data in [("AllReduce", ar), ("AllGather", ag), ("ReduceScatter", rs)]:
        if data and data["data"]:
            max_bw = max(d["busbw_gbps"] for d in data["data"])
            print(f"{name}: max busbw = {max_bw:.2f} GB/s")


if __name__ == "__main__":
    main()
