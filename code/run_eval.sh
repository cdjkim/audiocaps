#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
num_gpus=1
checkpoint_dir="./checkpoints/audiocaps/PyramidLSTM/audiocaps_pretrained"
ckpt="model.ckpt"

python eval.py \
    --num_gpus $num_gpus \
    --checkpoint_dir $checkpoint_dir \
    --ckpt $ckpt
