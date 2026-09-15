
# Whisper Fine-tuning with DeepSpeed ZeRO-Offload

Fine-tuning Whisper for Russian ASR on a single consumer GPU (RTX 5080 Laptop,
16GB VRAM) using DeepSpeed ZeRO Stage 3 with CPU offload.

## Motivation

Whisper models can exceed the VRAM of a single consumer GPU during full
fine-tuning. This project demonstrates how DeepSpeed ZeRO with CPU offload
makes it possible to train a large audio model on one GPU — the same
sharding mechanics used in multi-GPU training, but with CPU RAM instead of
additional GPUs.

The goal was to build a reproducible ASR fine-tuning pipeline and compare
several training configurations (baseline, frozen encoder, ZeRO-1/2/3 with
default and tuned communication buffers) in terms of VRAM usage, CPU RAM
usage, training time, and WER.

## Results

Training on 270 Russian Common Voice samples, Whisper-small, 3 epochs.
Evaluation on 30 held-out samples. WER measured with `evaluate` (jiwer).
All fine-tuning runs use `freeze_encoder()` and `optim="adamw_torch"`.

| Configuration | Max VRAM (GB) | Peak CPU RAM (GB) | Time | WER |
|---|---|---|---|---|
| Whisper-small, pretrained (no fine-tuning) | — | — | — | 28.14% |
| Baseline, 3 epochs | 5.94 | — | 50 s | 31.18% |
| Baseline + frozen encoder, 3 epochs | 3.98 | 9.07 | 32 s | 30.80% |
| ZeRO-1 + offload, 3 epochs | 2.17 | 12.32 | 41 s | 29.66% |
| ZeRO-2 + offload (default buffers), 3 epochs | 3.04 | 12.57 | 58 s | 27.38% |
| **ZeRO-2 + offload (tuned buffers), 3 epochs** | **1.21** | **12.60** | **50 s** | **25.86%** |
| ZeRO-3 + offload (default buffers), 3 epochs | 1.99 | 13.29 | 102 s | 25.10% |
| **ZeRO-3 + offload (tuned buffers), 3 epochs** | **0.85** | **13.30** | **86 s** | **28.52%** |

**Key findings:**

- **Best VRAM overall: 0.85 GB** — ZeRO-3 with tuned buffers is the
  absolute minimum, 7× less than baseline (5.94 GB).
- **Best balance: ZeRO-2 tuned** — 1.21 GB VRAM, 25.86% WER, 50 s,
  12.60 GB RAM. It gives the best quality-per-GB among offload configs.
- **Buffer tuning matters more than the ZeRO stage itself.**
  - ZeRO-2: 3.04 → 1.21 GB (−60%) by reducing `reduce_bucket_size` and
    `allgather_bucket_size` from 5e8 to 2e7.
  - ZeRO-3: 1.99 → 0.85 GB (−57%) by reducing
    `stage3_prefetch_bucket_size` from 5e8 to 5e7.
- **Tuning can hurt quality.** ZeRO-3 tuned saved 57% VRAM but worsened
  WER from 25.10% to 28.52% — likely because smaller prefetch buffers
  slow down parameter access and hurt convergence.
- **Offload shifts ~2.8 GB from VRAM to CPU RAM:** baseline + frozen
  uses 3.98 GB VRAM / 9.07 GB RAM, while ZeRO-2 tuned uses 1.21 GB VRAM
  / 12.60 GB RAM.
- **Catastrophic forgetting** in baseline: WER worsens from 28.14%
  (pretrained) to 38.40% (1 epoch). Freezing the encoder fixes this.
- **Gradient stability:** DeepSpeedCPUAdam produces stable gradients
  (grad_norm ~2–15) compared to FusedAdam in baseline (grad_norm ~98000).

### Tradeoff summary

| Goal | Best configuration | VRAM | WER | Time |
|---|---|---|---|---|
| Minimum VRAM | ZeRO-3 tuned | 0.85 GB | 28.52% | 86 s |
| Best quality/GB | ZeRO-2 tuned | 1.21 GB | 25.86% | 50 s |
| Best WER | ZeRO-3 default | 1.99 GB | 25.10% | 102 s |
| Fastest | Baseline + frozen | 3.98 GB | 30.80% | 32 s |

### Visual comparison

| WER | VRAM |
|---|---|
| ![WER](results/figures/wer_comparison.png) | ![VRAM](results/figures/vram_comparison.png) |

| CPU RAM | Time |
|---|---|
| ![RAM](results/figures/ram_comparison.png) | ![Time](results/figures/time_comparison.png) |

**Tradeoffs:**

| VRAM vs WER | VRAM vs CPU RAM |
|---|---|
| ![Tradeoff](results/figures/tradeoff.png) | ![Memory shift](results/figures/tradeoff_memory.png) |

### ZeRO stage comparison

![ZeRO stages](results/figures/zero_stages.png)

### Effect of buffer tuning

![Buffer tuning](results/figures/buffer_tuning.png)

## Hardware

- GPU: NVIDIA GeForce RTX 5080 Laptop (16GB VRAM, sm_120, Blackwell)
- CPU RAM: 32GB+
- OS: Linux (Ubuntu)
- CUDA: 12.8 (system nvcc 12.9, minor mismatch tolerated via
  `DS_SKIP_CUDA_CHECK=1`)

## Stack

- PyTorch 2.12.0.dev+cu128 (nightly, sm_120 support)
- torchaudio
- HuggingFace Transformers 4.51.3, Datasets, Evaluate, Accelerate 0.29.3
- DeepSpeed (ZeRO Stage 1/2/3 + CPU offload)
- soundfile, librosa for audio I/O
- DVC for data versioning
- TensorBoard
- Poetry for dependency management

## Project structure
```
whisper-finetuning/
├── configs/
│   ├── ds_baseline.json              # DeepSpeed without offload
│   ├── ds_zero1_offload.json         # ZeRO-1 + CPU offload
│   ├── ds_zero2_offload.json         # ZeRO-2 + CPU offload (tuned buffers)
│   ├── ds_zero2_offload_default.json # ZeRO-2 + CPU offload (default buffers)
│   ├── ds_zero3_offload.json         # ZeRO-3 + CPU offload (default buffers)
│   └── ds_zero3_offload_tuned.json   # ZeRO-3 + CPU offload (tuned buffers)
├── src/
│   ├── preprocess.py            # Audio preprocessing + manifest creation
│   ├── dataset.py               # WhisperDataset + DataCollator
│   ├── train.py                 # Training script
│   ├── run_eval.py              # WER evaluation
│   ├── eval_pretrained.py       # WER of pretrained model (baseline)
│   └── utils.py                 # Logging, seed, VRAM/CPU RAM tracking
├── scripts/
│   ├── download_common_voice.py # Download Common Voice ru subset
│   └── make_plots.py            # Generate comparison plots
├── data/                        # DVC-tracked (audio + manifests)
├── raw_audio/                   # Original WAV files
├── results/                     # Comparison tables, plots, predictions
│   └── figures/                 # Generated plots for README
└── pyproject.toml
```

## Data

**Dataset:** Common Voice 17.0 (Russian), via HuggingFace mirror
`fsicoli/common_voice_17_0`.

**Size:** 300 audio clips (16kHz, mono, WAV), split into 270 train / 30 eval.

**Pipeline:**

1. Download a subset via `scripts/download_common_voice.py`.
2. Preprocess audio to 16kHz mono WAV and create a JSONL manifest.
3. Version data with DVC.

**Manifest format (JSONL):**

```json
{"audio": "data/processed_audio/cv_00000.wav", "text": "транскрипция"}
```

## Setup

```bash
# Install dependencies
poetry install

# Install PyTorch with CUDA 12.8 (sm_120 support)
poetry run pip uninstall torch torchaudio -y
poetry run pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Pin compatible transformers/accelerate versions
poetry run pip install "transformers==4.51.3" "accelerate==0.29.3"

# Install DeepSpeed
poetry run pip install deepspeed

# Verify GPU
poetry run python -c "import torch; print(torch.cuda.get_arch_list())"
# Expected: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

## Data preparation

```bash
# Download Common Voice ru subset (requires HuggingFace auth)
poetry run huggingface-cli login
poetry run python scripts/download_common_voice.py

# Preprocess audio + create manifest
poetry run python -m src.preprocess \
    --input raw_audio/ \
    --output data/processed_audio/ \
    --manifest data/train_manifest.jsonl \
    --transcriptions raw_transcriptions.json

# Split into train/eval (90/10)
poetry run python -c "
import json, random
random.seed(42)
with open('data/train_manifest.jsonl', encoding='utf-8') as f:
    items = [json.loads(l) for l in f if l.strip()]
random.shuffle(items)
split = int(len(items) * 0.9)
with open('data/train_manifest.jsonl', 'w', encoding='utf-8') as f:
    for it in items[:split]:
        f.write(json.dumps(it, ensure_ascii=False) + '\n')
with open('data/eval_manifest.jsonl', 'w', encoding='utf-8') as f:
    for it in items[split:]:
        f.write(json.dumps(it, ensure_ascii=False) + '\n')
"

# Track with DVC
poetry run dvc init
poetry run dvc add data/
git add .dvc .dvcignore data.dvc data/.gitignore
git commit -m "Track data with DVC"
```

## Training

**Baseline (no offload):**

```bash
PYTHONPATH=. poetry run deepspeed --num_gpus=1 src/train.py \
    --deepspeed configs/ds_baseline.json \
    --output_dir checkpoints/baseline \
    --num_epochs 3
```

**ZeRO-1 / ZeRO-2 / ZeRO-3 (with offload):**

```bash
# ZeRO-1
DS_SKIP_CUDA_CHECK=1 PYTHONPATH=. poetry run deepspeed --num_gpus=1 src/train.py \
    --deepspeed configs/ds_zero1_offload.json \
    --output_dir checkpoints/zero1_offload \
    --num_epochs 3

# ZeRO-2 (tuned buffers)
DS_SKIP_CUDA_CHECK=1 PYTHONPATH=. poetry run deepspeed --num_gpus=1 src/train.py \
    --deepspeed configs/ds_zero2_offload.json \
    --output_dir checkpoints/zero2_offload \
    --num_epochs 3

# ZeRO-3 (tuned buffers)
DS_SKIP_CUDA_CHECK=1 PYTHONPATH=. poetry run deepspeed --num_gpus=1 src/train.py \
    --deepspeed configs/ds_zero3_offload_tuned.json \
    --output_dir checkpoints/zero3_offload_tuned \
    --num_epochs 3
```

**Note:** All configs use `optim="adamw_torch"` (standard PyTorch AdamW)
instead of DeepSpeed FusedAdam due to a JIT compilation bug on Blackwell
GPUs. See Notes and workarounds #8. The encoder is frozen via
`model.freeze_encoder()` to prevent catastrophic forgetting on small
datasets.

## Evaluation

**Pretrained model (baseline):**

```bash
poetry run python -m src.eval_pretrained \
    --manifest data/eval_manifest.jsonl \
    --language russian \
    --limit 30
```

**Fine-tuned model:**

```bash
poetry run python -m src.run_eval \
    --model checkpoints/zero2_offload \
    --manifest data/eval_manifest.jsonl \
    --language russian \
    --limit 30
```

**ZeRO-3 checkpoint:** DeepSpeed saves the model as sharded checkpoints.
To evaluate with HuggingFace `from_pretrained`, gather weights first:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 poetry run python checkpoints/zero3_offload_tuned/zero_to_fp32.py \
    checkpoints/zero3_offload_tuned \
    checkpoints/zero3_offload_tuned/pytorch_model.bin
```

## Plots

```bash
poetry run python scripts/make_plots.py
```

Generates 8 comparison plots in `results/figures/`.

## What I learned

- How ZeRO Stage 1/2/3 shard parameters, gradients, and optimizer states.
- How CPU offload enables training beyond VRAM limits, at the cost of
  CPU RAM and throughput.
- Why `overlap_comm` and `contiguous_gradients` matter for throughput.
- How the same sharding mechanics scale to multi-GPU training.
- Audio preprocessing specifics for Whisper: 16kHz mono, log-mel
  spectrogram, 3000-frame padding.
- WER as the primary metric for ASR.
- **Buffer tuning matters more than the ZeRO stage itself.** Reducing
  `reduce_bucket_size`, `allgather_bucket_size`, and
  `stage3_prefetch_bucket_size` from 5e8 to 2e7/5e7 cuts VRAM by 57–60%
  with no speed penalty (and sometimes a speedup).
- **Tuning can hurt quality.** ZeRO-3 tuned saved 57% VRAM but worsened
  WER from 25.10% to 28.52% — smaller prefetch buffers slow down parameter
  access and hurt convergence.
- **Best balance is ZeRO-2 tuned:** 1.21 GB VRAM, 25.86% WER, 50 s.
- **Offload is not free:** it shifts ~2.8 GB from VRAM to CPU RAM and
  slows down training by 1.5–3×.
- **Catastrophic forgetting** on small datasets: baseline WER grew from
  28.14% to 38.40% after 1 epoch. Freezing the encoder fixes this.
- **Gradient stability:** DeepSpeedCPUAdam produces stable gradients
  (grad_norm ~2–15) compared to FusedAdam in baseline (grad_norm ~98000).

## Notes and workarounds

This project required several non-obvious workarounds due to running on
brand-new hardware (RTX 5080, sm_120) with nightly PyTorch:

1. **PyTorch nightly with CUDA 12.8** is required — stable PyTorch 2.5.x
   with CUDA 12.1 does not include sm_120 kernels, causing
   `no kernel image is available` at runtime.

2. **`torchaudio.save/load` replaced with `soundfile` + `librosa`** to avoid
   the `torchcodec` / FFmpeg dependency chain. `torchcodec` requires system
   FFmpeg libraries (`libavutil.so.xx`) that may not be present.

3. **`src/evaluate.py` renamed to `src/run_eval.py`** to avoid a name
   collision with the PyPI `evaluate` library, which caused `train.py` to
   import the local file instead of the library.

4. **`--local_rank` added to argparse** because DeepSpeed passes this
   argument automatically to the training script.

5. **`train_micro_batch_size_per_gpu: "auto"`** in DeepSpeed configs to fix
   `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` in
   `accelerate`.

6. **`TORCH_CUDA_ARCH_LIST="12.0"`** set explicitly to prevent DeepSpeed's
   JIT compiler from generating an invalid `compute_1.` architecture flag
   when building `fused_adam`.

7. **`transformers==4.51.3` and `accelerate==0.29.3`** pinned to avoid
   `TypeError: Accelerator.unwrap_model() got an unexpected keyword argument
   'keep_torch_compile'`.

8. **FusedAdam JIT compilation fails on Blackwell (sm_120).** DeepSpeed's
   `FusedAdam` optimizer triggers JIT compilation of a CUDA extension
   (`fused_adam`) on first use. On Blackwell GPUs (sm_120), DeepSpeed
   generates an invalid architecture flag `compute_1.` instead of
   `compute_100`, causing `nvcc fatal: Unsupported gpu architecture
   'compute_1.'`. **Workaround:** disable `FusedAdam` by setting
   `optim="adamw_torch"` in `Seq2SeqTrainingArguments` and removing the
   `"optimizer"` section from the DeepSpeed configs.

9. **CUDA version mismatch for ZeRO-Offload.** DeepSpeed's `DeepSpeedCPUAdam`
   (used automatically by `accelerate` for ZeRO-Offload) requires JIT
   compilation and checks that the system CUDA toolkit version matches the
   version PyTorch was compiled with. With system CUDA 12.9 and PyTorch
   compiled with CUDA 12.8, this causes `CUDAMismatchException`.
   **Workaround:** set `DS_SKIP_CUDA_CHECK=1` when running the offload
   training.

10. **ZeRO-3 checkpoint format.** With ZeRO Stage 3, DeepSpeed saves the
    model as sharded checkpoints, not as a standard `pytorch_model.bin`.
    To evaluate with HuggingFace `from_pretrained`, run `zero_to_fp32.py`
    to gather weights into a single file.

11. **`torch.load` weights_only error for DeepSpeed checkpoints.** PyTorch
    2.6+ changed the default of `weights_only` to `True`, which prevents
    loading DeepSpeed optimizer states containing custom classes like
    `ZeroStageEnum`. **Workaround:** set
    `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` when running `zero_to_fp32.py`.

12. **`forced_decoder_ids` error in transformers 4.50+.** After fine-tuning,
    the saved Whisper checkpoint contains `forced_decoder_ids` in its
    generation config. With transformers 4.50+, this attribute was deprecated
    and causes `ValueError: You have explicitly specified forced_decoder_ids`.
    **Workaround:** set `model.generation_config.forced_decoder_ids = None`
    after loading the model for evaluation.

These workarounds are documented for reproducibility and as a reference for
others running modern ML workloads on Blackwell consumer GPUs.

## Limitations

- **Small dataset:** 270 training samples (~30 minutes of audio) is not
  enough to fully fine-tune Whisper. WER of 25.10% is better than
  pretrained (28.14%), but far from production quality (~10%).
- **Single GPU:** The project demonstrates ZeRO sharding between GPU and
  CPU, not between multiple GPUs. The mechanics are the same, but actual
  multi-GPU scaling was not tested.
- **No audio/video multimodality:** Only audio (ASR) is covered. Video and
  multimodal training are out of scope.
- **Whisper-small only:** Larger variants (medium, large-v3) were not
  tested due to VRAM constraints.
