"""Сравнение Gloo vs NCCL на all_reduce."""

import os
import time
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def benchmark_backend(backend: str, rank: int, world_size: int,
                       size_mb: int = 64, num_iters: int = 50):
    """Бенчмарк all_reduce для заданного backend."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29510"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    num_elements = size_mb * 1024 * 1024 // 4
    tensor = torch.randn(num_elements, device=device)

    for _ in range(5):
        dist.all_reduce(tensor)

    if backend == "nccl":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        dist.all_reduce(tensor)
    if backend == "nccl":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    busbw = 2 * (world_size - 1) / world_size * tensor.nbytes * num_iters / elapsed / 1e9

    dist.destroy_process_group()
    return busbw


# === TOP-LEVEL WORKERS (не внутри main) ===

def nccl_worker(rank, world_size, size_mb, return_dict):
    bw = benchmark_backend("nccl", rank, world_size, size_mb)
    if rank == 0:
        return_dict["nccl"] = bw


def gloo_worker(rank, world_size, size_mb, return_dict):
    bw = benchmark_backend("gloo", rank, world_size, size_mb)
    if rank == 0:
        return_dict["gloo"] = bw


def main():
    results = {}
    size_mb = 64

    # NCCL
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(nccl_worker, args=(world_size, size_mb, return_dict),
                 nprocs=world_size, join=True)
        results["nccl"] = return_dict.get("nccl", 0)
        print(f"NCCL all_reduce ({size_mb} MB): {results['nccl']:.2f} GB/s")

    # Gloo (CPU)
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(gloo_worker, args=(2, size_mb, return_dict), nprocs=2, join=True)
    results["gloo"] = return_dict.get("gloo", 0)
    print(f"Gloo all_reduce ({size_mb} MB): {results['gloo']:.2f} GB/s")

    with open("results/gloo_vs_nccl.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
