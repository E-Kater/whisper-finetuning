"""Препроцессинг аудио: приведение к 16kHz, моно, WAV + создание манифеста."""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger(__name__)

TARGET_SR = 16000
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def preprocess_audio(input_dir: str, output_dir: str, target_sr: int = TARGET_SR) -> None:
    """Конвертирует все аудио в 16kHz моно WAV."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_path.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    logger.info(f"Найдено {len(files)} аудиофайлов в {input_dir}")

    for fname in tqdm(files, desc="Preprocessing"):
        try:
            audio_np, sr = sf.read(str(fname), dtype="float32")
        except Exception as e:
            logger.warning(f"Не удалось загрузить {fname}: {e}")
            continue

        # Моно
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Ресемплинг
        if sr != target_sr:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)

        out_file = output_path / (fname.stem + ".wav")
        sf.write(str(out_file), audio_np, target_sr)

    logger.info(f"Готово. Файлы в {output_dir}")


def create_manifest(
    audio_dir: str,
    output_jsonl: str,
    transcriptions: dict | None = None,
) -> None:
    """
    Создаёт JSONL-манифест.
    Формат: {"audio": "...", "text": "..."}
    """
    audio_path = Path(audio_dir)
    files = sorted([f for f in audio_path.iterdir() if f.suffix.lower() == ".wav"])
    logger.info(f"Создаю манифест для {len(files)} файлов")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for fname in files:
            text = ""
            if transcriptions and fname.stem in transcriptions:
                text = transcriptions[fname.stem]
            elif transcriptions is None:
                text = f"[PLACEHOLDER] {fname.stem}"
            entry = {"audio": str(fname), "text": text}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Манифест сохранён: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Препроцессинг аудио для Whisper")
    parser.add_argument("--input", required=True, help="Папка с исходным аудио")
    parser.add_argument("--output", required=True, help="Папка для обработанного аудио")
    parser.add_argument("--manifest", required=True, help="Путь для JSONL-манифеста")
    parser.add_argument("--transcriptions", default=None, help="JSON с транскрипциями {id: text}")
    args = parser.parse_args()

    preprocess_audio(args.input, args.output)

    transcriptions = None
    if args.transcriptions:
        with open(args.transcriptions, encoding="utf-8") as f:
            transcriptions = json.load(f)

    create_manifest(args.output, args.manifest, transcriptions)


if __name__ == "__main__":
    main()