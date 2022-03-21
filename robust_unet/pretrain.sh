#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py --arch transunet --aug-strength 8 --batch-size 12 --contrastive byol \
    --cos --dataset Synapse --epochs 100 --lr 0.2 --momentum-encoder 0.99 --num-gpu 1 --num-workers 8 --wd 0.0001 
CUDA_VISIBLE_DEVICES=0 python pretrain.py --arch swinunet --aug-strength 8 --batch-size 12 --contrastive byol \
    --cos --dataset Synapse --epochs 100 --lr 0.2 --momentum-encoder 0.99 --num-gpu 1 --num-workers 8 --wd 0.0001 
CUDA_VISIBLE_DEVICES=0 python pretrain.py --arch transunet --aug-strength 16 --batch-size 12 --contrastive byol \
    --cos --dataset Synapse --epochs 100 --lr 0.2 --momentum-encoder 0.99 --num-gpu 1 --num-workers 8 --wd 0.0001 
CUDA_VISIBLE_DEVICES=0 python pretrain.py --arch swinunet --aug-strength 16 --batch-size 12 --contrastive byol \
    --cos --dataset Synapse --epochs 100 --lr 0.2 --momentum-encoder 0.99 --num-gpu 1 --num-workers 8 --wd 0.0001 