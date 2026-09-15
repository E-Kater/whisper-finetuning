import json
from typing import Any

import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset
from transformers import WhisperProcessor


class WhisperDataset(Dataset):
    def __init__(self, manifest_path: str, processor: WhisperProcessor):
        self.processor = processor
        with open(manifest_path, encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        audio_np, sr = sf.read(item["audio"], dtype="float32")

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        features = self.processor.feature_extractor(
            audio_np, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(item["text"]).input_ids
        return {"input_features": features, "labels": labels}


class DataCollatorWhisper:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch