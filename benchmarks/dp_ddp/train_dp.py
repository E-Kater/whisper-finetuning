"""DP: single-process data parallelism (master GPU bottleneck)."""

import time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP, count_params


def main():
    device = torch.device("cuda:0")
    model = MLP().to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])

    print(f"DP | params: {count_params(model.module) / 1e6:.2f}M")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    total_samples = 0
    losses = []
    accs = []

    for epoch in range(10):
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
        print(f"Epoch {epoch} | loss: {loss.item():.4f} | acc: {acc:.4f}")

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    result = {
        "strategy": "DP",
        "model": "MLP",
        "params_m": count_params(model.module) / 1e6,
        "world_size": 2,
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
    with open("results/results_dp.json", "w") as f:
        json.dump(result, f)
    print(f"DP | time={elapsed:.1f}s | throughput={total_samples/elapsed:.1f} samples/s | peak_mem={peak_mem:.2f} GB | acc={accs[-1]:.4f}")


if __name__ == "__main__":
    main()
