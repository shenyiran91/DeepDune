CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 150 --base_lr 0.01 --load_from contrastive --decay_lr 1

CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 150 --base_lr 0.01 --load_from contrastive --decay_lr 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 150 --base_lr 0.01 --decay_lr 1

CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --momentum 0.9 --max_epochs 150 --base_lr 0.01 --decay_lr 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150  --base_lr 0.01 --lam 0.2 --load_from contrastive

CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150  --base_lr 0.01 --lam 0.2 --load_from contrastive

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150  --base_lr 0.01 --lam 0.2

CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --loss ada_image --momentum 0.9 --max_epochs 150  --base_lr 0.01 --lam 0.2

