"""Оценка WER для обученной модели Whisper."""

import argparse
import json

import evaluate
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.utils import get_logger

logger = get_logger(__name__)

TARGET_SR = 16000


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Загружает аудио, приводит к моно и target_sr."""
    audio_np, sr = sf.read(path, dtype="float32")

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    if sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)

    return audio_np


def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    language: str = "russian",
) -> str:
    # Фикс для forced_decoder_ids (transformers 4.50+)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config._from_model_config = True
    
    """Транскрибирует один аудиофайл."""
    audio_np = load_audio(audio_path)

    inputs = processor.feature_extractor(
        audio_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).input_features

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            language=language,
            task="transcribe",
        )

    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Оценка WER для Whisper")
    parser.add_argument("--model", required=True, help="Путь к обученной модели")
    parser.add_argument("--manifest", required=True, help="JSONL с аудио и текстом")
    parser.add_argument("--language", default="russian")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число примеров")
    parser.add_argument("--output", default="results/predictions.jsonl")
    args = parser.parse_args()

    logger.info(f"Загружаю модель: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()

    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        items = items[: args.limit]

    logger.info(f"Оцениваю {len(items)} примеров...")

    predictions, references = [], []
    for i, item in enumerate(items):
        pred = transcribe(model, processor, item["audio"], args.language)
        predictions.append(pred)
        references.append(item["text"])
        if (i + 1) % 10 == 0:
            logger.info(f"Обработано {i + 1}/{len(items)}")

    wer_metric = evaluate.load("wer")
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    logger.info(f"WER: {wer:.2f}%")

    # Сохраняем примеры предсказаний
    with open(args.output, "w", encoding="utf-8") as f:
        for pred, ref in zip(predictions, references):
            f.write(json.dumps({"pred": pred, "ref": ref}, ensure_ascii=False) + "\n")

    logger.info(f"Предсказания сохранены: {args.output}")


if __name__ == "__main__":
    main()