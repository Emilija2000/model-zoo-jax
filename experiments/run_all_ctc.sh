#!/bin/bash
set -o errexit

for task in batch_size augmentation optimizer activation initialization
do
    python train_ctc.py --task $task --use_wandb
done
