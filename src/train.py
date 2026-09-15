"""Скрипт обучения Whisper с DeepSpeed ZeRO-3 offload."""

import argparse
from dataclasses import dataclass

import gc
import psutil
import evaluate
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.dataset import WhisperDataset, DataCollatorWhisper
from src.utils import get_logger, log_gpu_memory, reset_gpu_peak, set_seed

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name: str = "openai/whisper-small"
    train_manifest: str = "data/train_manifest.jsonl"
    eval_manifest: str = "data/eval_manifest.jsonl"
    output_dir: str = "checkpoints/whisper-finetune"
    language: str = "russian"
    task: str = "transcribe"
    num_epochs: int = 3
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    deepspeed_config: str = "configs/ds_zero3_offload.json"
    fp16: bool = True
    seed: int = 42


def log_cpu_memory(prefix: str = "", peak_tracker: dict | None = None) -> float:
    """Замеряет текущее и пиковое потребление CPU RAM в GB.

    Если peak_tracker передан, пик хранится в нём (чтобы не терять
    между вызовами). Иначе используется атрибут функции.
    """
    gc.collect()
    vm = psutil.virtual_memory()
    used_gb = (vm.total - vm.available) / (1024 ** 3)

    if peak_tracker is not None:
        peak_tracker["peak"] = max(peak_tracker.get("peak", 0.0), used_gb)
        peak_used = peak_tracker["peak"]
    else:
        peak_used = max(getattr(log_cpu_memory, "_peak", 0.0), used_gb)
        log_cpu_memory._peak = peak_used

    if prefix:
        print(f"[{prefix}] CPU RAM used: {used_gb:.2f} GB | peak: {peak_used:.2f} GB")
    return used_gb


def build_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=TrainConfig.model_name)
    parser.add_argument("--train_manifest", default=TrainConfig.train_manifest)
    parser.add_argument("--eval_manifest", default=TrainConfig.eval_manifest)
    parser.add_argument("--output_dir", default=TrainConfig.output_dir)
    parser.add_argument("--language", default=TrainConfig.language)
    parser.add_argument("--num_epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--deepspeed", default=TrainConfig.deepspeed_config)
    parser.add_argument("--fp16", action="store_true", default=TrainConfig.fp16)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    set_seed(TrainConfig.seed)
    reset_gpu_peak()

    peak_tracker = {"peak": 0.0}

    logger.info(f"Загружаю процессор: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task="transcribe"
    )

    logger.info(f"Загружаю модель: {args.model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.use_cache = False
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.freeze_encoder()

    log_cpu_memory("after model load", peak_tracker)

    train_ds = WhisperDataset(args.train_manifest, processor)
    eval_ds = WhisperDataset(args.eval_manifest, processor)
    collator = DataCollatorWhisper(processor)

    logger.info(f"Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=TrainConfig.per_device_batch_size,
        gradient_accumulation_steps=TrainConfig.grad_accum_steps,
        learning_rate=TrainConfig.learning_rate,
        max_grad_norm=TrainConfig.max_grad_norm,
        warmup_steps=TrainConfig.warmup_steps,
        num_train_epochs=args.num_epochs,
        eval_strategy="steps",
        eval_steps=TrainConfig.eval_steps,
        save_steps=TrainConfig.save_steps,
        logging_steps=TrainConfig.logging_steps,
        predict_with_generate=True,
        fp16=args.fp16,
        optim="adamw_torch",
        report_to=["tensorboard"],
        deepspeed=args.deepspeed,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
    )

    logger.info("Старт обучения...")
    log_cpu_memory("before training", peak_tracker)
    trainer.train()

    log_gpu_memory("after training")
    log_cpu_memory("after training", peak_tracker)

    logger.info(f"Сохраняю модель в {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info("Готово.")


if __name__ == "__main__":
    main()