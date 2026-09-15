#!/bin/bash
set -e

poetry run deepspeed --num_gpus=1 src/train.py \
    --deepspeed configs/ds_baseline.json \
    --output_dir checkpoints/baseline \
    --num_epochs 3