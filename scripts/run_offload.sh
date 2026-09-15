#!/bin/bash
set -e

poetry run deepspeed --num_gpus=1 src/train.py \
    --deepspeed configs/ds_zero3_offload.json \
    --output_dir checkpoints/zero3_offload \
    --num_epochs 3