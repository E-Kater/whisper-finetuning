"""Оценка WER для оригинальной (не дообученной) модели Whisper."""

import argparse
import json

import evaluate
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.run_eval import load_audio
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", default="russian")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()][: args.limit]

    preds, refs = [], []
    for item in items:
        audio_np = load_audio(item["audio"])
        inputs = processor.feature_extractor(
            audio_np, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inputs, language=args.language, task="transcribe")
        preds.append(processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0])
        refs.append(item["text"])

    wer = 100 * evaluate.load("wer").compute(predictions=preds, references=refs)
    logger.info(f"Pretrained WER: {wer:.2f}%")


if __name__ == "__main__":
    main()