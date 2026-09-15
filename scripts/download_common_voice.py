"""Скачивает Common Voice ru и конвертирует в формат проекта."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("raw_audio")
TRANSCRIPTIONS_FILE = Path("raw_transcriptions.json")

NUM_SAMPLES = 300  # сколько примеров взять
TARGET_SR = 16000


def main():
    RAW_DIR.mkdir(exist_ok=True)

    print(f"Скачиваю {NUM_SAMPLES} примеров Common Voice ru...")
    ds = load_dataset(
        "fsicoli/common_voice_17_0",
        "ru",
        split=f"train[:{NUM_SAMPLES}]",
        trust_remote_code=True,
    )

    transcriptions = {}
    for i, item in enumerate(tqdm(ds, desc="Saving audio")):
        audio = item["audio"]
        audio_np = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        # Моно: если 2 канала — усредняем
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Ресемплинг к 16kHz через librosa
        if sr != TARGET_SR:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=TARGET_SR)

        fname = f"cv_{i:05d}.wav"
        out_path = RAW_DIR / fname
        sf.write(str(out_path), audio_np, TARGET_SR)

        transcriptions[out_path.stem] = item["sentence"]

    with open(TRANSCRIPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    print(f"Готово: {len(transcriptions)} файлов в {RAW_DIR}")
    print(f"Транскрипции: {TRANSCRIPTIONS_FILE}")


if __name__ == "__main__":
    main()