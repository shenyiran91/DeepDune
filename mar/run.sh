#!/usr/bin/env bash

# epochs = 50; batch size = 32;

# Choice of LOSS

# Choice of MODEL

for MODEL in Unet Unetpp TransUnet;
do
    for LOSS in cross_entropy comb_loss focal_loss balanced_ce;
    do
        for LR in 0.001 0.01 0.1
        do
            for WD in 1e-5 1e-4 1e-3 0
            do
                python3 ./train.py --epochs 50 --model $MODEL --loss $LOSS --lr $LR --bs 32 --opt adam --save_model
            done
        done
    done
done