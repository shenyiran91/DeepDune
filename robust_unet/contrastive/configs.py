from datetime import datetime
import os, sys
import ml_collections
import argparse
import math

def get_argin(arch='transunet', contrastive='simsiam', dataset='Synapse', ): # for notebook
    parser = argparse.ArgumentParser()
    cfg = parser.parse_args('')
    cfg.arch = arch
    cfg.contrastive = contrastive
    cfg.dataset = dataset
    cfg.lr = 0.2
    cfg.epochs = 100
    cfg.schedule = [60, 80]
    cfg.cos = True

    cfg.batch_size = 24
    cfg.wd = 1e-4

    # moco/simsiam specific cfgs:
    cfg.img_size = 224
    cfg.feature_dim = 768
    cfg.pred_dim = 256 # for simsiam
    cfg.aug_strength = 10
    cfg.momentum_encoder = 0.99
    # for moco
    cfg.moco_k = 128
    cfg.moco_m = 0.99
    cfg.moco_t = 0.1
    cfg.symmetric = True

    # resources
    cfg.num_gpu = 1
    cfg.num_workers = 4

    # utils
    cfg.resume = ''
    cfg.results_dir = ''

    return cfg

def adjust_lr_schedule(config):
    if config.schedule[-1]>config.epochs:
        config.schedule = [math.floor(config.epochs*0.6), math.floor(config.epochs*0.8)]

def get_basic_config(argin):
    config = ml_collections.ConfigDict(vars(argin))
    # trainer
    config.lr = config.lr * config.batch_size / 256.
    if config.cos:
        config.schedule = []  # cos in use
    else: 
        adjust_lr_schedule(config)
    # utils
    if config.results_dir == '':
        if config.cos:
            lr_schedule = 'cos'
        elif len(config.schedule) > 0:
            lr_schedule = 'step{:d}'.format(len(config.schedule))
        else:
            lr_schedule = 'fixed'
        config.results_dir = '../results/_{:s}-{:s}-{:s}_aug{:d}_bsz{:d}_epo{:d}_lr{:.0e}_lrs-{:s}_wd{:.0e}_'.format(
            config.contrastive, config.arch, config.dataset, config.aug_strength, config.batch_size,
            config.epochs, config.lr, lr_schedule, config.wd) + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config.use_checkpoint = False
    return config

def get_synapse_config(config):
    config.num_classes = 9
    config.synapse_list_dir = './lists/lists_Synapse'
    config.synapse_base_dir = '../data/Synapse/train_npz'
    return config

def get_swinunet_config(config): 
    config.arch = 'swinunet'
    config.ATTN_DROP_RATE = 0.
    config.PATCH_SIZE = 4
    config.IN_CHANS = 3
    config.EMBED_DIM = 96
    config.DEPTHS = [2, 2, 2, 2]
    config.NUM_HEADS = [3, 6, 12, 24]
    config.WINDOW_SIZE = 7
    config.MLP_RATIO = 4.
    config.QKV_BIAS = True
    config.QK_SCALE = None
    config.DROP_RATE = 0.
    config.DROP_PATH_RATE = 0.2
    config.APE = False
    config.PATCH_NORM = True
    config.PRETRAIN_CKPT = '../pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
    return config 

def get_transunet_config(config):
    config.arch = 'transunet'
    # get_b16_configs
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    
    # get_r50_b16_config
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    if config.pretrained_path.find('R50') != -1:
        config.patches.grid = (int(config.img_size / config.patch_size), int(config.img_size / config.patch_size))
    return config

