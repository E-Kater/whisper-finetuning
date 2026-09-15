#!/bin/bash
set -e

# Пример: скачать Common Voice через HuggingFace datasets
# Требуется авторизация: huggingface-cli login

poetry run python -c "
from datasets import load_dataset
import os

os.makedirs('data', exist_ok=True)
ds = load_dataset('mozilla-foundation/common_voice_16_1', 'ru', split='train[:1000]')
ds.save_to_disk('data/common_voice_ru')
print('Saved to data/common_voice_ru')
"

echo "Данные скачаны. Дальше нужен скрипт конвертации в JSONL."