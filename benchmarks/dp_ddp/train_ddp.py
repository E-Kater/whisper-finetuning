"""DDP: multi-process data parallelism with NCCL all-reduce."""

import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model import MLP, count_params


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Model
    model = MLP().to(device)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(f"DDP | params: {count_params(model.module) / 1e6:.2f}M")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    total_samples = 0
    losses = []
    accs = []

    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_samples += data.size(0)
            losses.append(loss.item())
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        acc = correct / total
        accs.append(acc)
        if rank == 0:
            print(f"Epoch {epoch} | loss: {loss.item():.4f} | acc: {acc:.4f}")

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        result = {
            "strategy": "DDP",
            "model": "MLP",
            "params_m": count_params(model.module) / 1e6,
            "world_size": world_size,
            "epochs": 10,
            "batch_size": 256,
            "total_time_s": elapsed,
            "throughput_samples_s": total_samples / elapsed,
            "peak_memory_gb": peak_mem,
            "final_loss": losses[-1],
            "final_acc": accs[-1],
            "losses": losses,
            "accs": accs,
        }
        with open("results/results_ddp.json", "w") as f:
            json.dump(result, f)
        print(f"DDP | time={elapsed:.1f}s | throughput={total_samples/elapsed:.1f} samples/s | peak_mem={peak_mem:.2f} GB | acc={accs[-1]:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
