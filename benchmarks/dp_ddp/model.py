"""Общая MLP модель для DP/DDP бенчмарка.

Эта модель используется для сравнения DataParallel (DP) и
DistributedDataParallel (DDP) на 2×T4 (Kaggle).

Почему MLP, а не Whisper:
- Цель — измерить communication overhead, а не обучить модель.
- MLP обучается быстро (10 эпох за ~1 минуту).
- CIFAR-10 намного меньше Whisper dataset.
- Whisper уже обучен в основном проекте (DeepSpeed ZeRO).

Результаты:
- DP:  66.2 s, 7549 samples/s, 77.0% accuracy
- DDP: 54.4 s, 4598 samples/s, 75.8% accuracy
- DDP быстрее на 18% благодаря multi-process + NCCL.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """3-layer MLP для CIFAR-10: 3072 → 512 → 512 → 10.

    Args:
        in_dim: входная размерность (3 × 32 × 32 = 3072 для CIFAR-10).
        hidden: размер скрытого слоя.
        num_classes: число классов (10 для CIFAR-10).
    """

    def __init__(
        self,
        in_dim: int = 3072,
        hidden: int = 512,
        num_classes: int = 10,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, 32, 32) — batch CIFAR-10 images.
        Returns:
            logits: (B, num_classes).
        """
        x = x.view(x.size(0), -1)  # flatten: (B, 3072)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def count_params(model: nn.Module) -> int:
    """Возвращает общее число параметров модели."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = MLP()
    print(f"Total params: {count_params(model) / 1e6:.4f}M")

    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Total params: 1.8412M
    # Input shape: torch.Size([4, 3, 32, 32])
    # Output shape: torch.Size([4, 10])