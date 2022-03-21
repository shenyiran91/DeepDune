#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150  --base_lr 0.01 --lam 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150  --base_lr 0.01 --lam 0.2
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150 --base_lr 0.01 --lam 0.2 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml'
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150 --base_lr 0.01 --lam 0.2 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml'
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 150 --base_lr 0.01 --decay_lr 1 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml' --dice_ratio 0.6
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 150 --base_lr 0.01 --decay_lr 1 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml' --dice_ratio 0.6
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150 --base_lr 0.05 --lam 0.2 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml' --dice_ratio 0.6
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ce_loss --momentum 0.9 --max_epochs 150 --base_lr 0.05 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml' --dice_ratio 0.6

CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150 --base_lr 0.05 --lam 0.2 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml' --dice_ratio 0.6


CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ce_loss --momentum 0.9 --max_epochs 150 --base_lr 0.05 --model swinunet --cfg './configs/swin_tiny_patch4_window7_224_lite.yaml' --dice_ratio 0.6


CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 5 --base_lr 0.01 --load_from contrastive