#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
BLUEFOG_LOG_LEVEL=debug BLUEFOG_WITH_NCCL=1 bfrun -np 2 python examples/pytorch_resnet.py \
        --epochs 100 \
        --batch-size 64 \
        --val-batch-size 64 \
        --base-lr 0.0125 \
        --dist-optimizer neighbor_allreduce \
        --seed 42 \