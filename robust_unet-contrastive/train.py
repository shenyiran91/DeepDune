import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

# from swin-unet
from config import get_config
from networks.vision_transformer import SwinUnet as Swin_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--loss', type=str, default='ce_loss',
                    help='loss type, default is ce_loss')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--decay_lr', type=int, default=0,
                    help='turn on decay learning rate with 1')
parser.add_argument('--lam', type=float, default=0.2,
                    help='lambda for adaptive method')
parser.add_argument('--model', type=str,
                    default='transunet', help='model network')
parser.add_argument('--dice_ratio', type=float,
                    default=0.5, help='ratio for dice loss')
parser.add_argument('--load_from', type=str, default='pretrained',
                    help='load model from pretrained or contrastive')
parser.add_argument('--contrastive_path', type=str, default='default', help='path for pretrained encoder using contrastive learning')
parser.add_argument('--corrupt_type', type=str)
parser.add_argument('--corrupt_ratio', type=float, help='label corruption ratio', default=0)

# from swin-unet
parser.add_argument('--cfg', type=str, metavar="FILE",
                    help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    args.exp = 'TU_' + dataset_name + '_' + \
        args.load_from + '_' + args.model + '_' + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + args.vit_name if args.model == 'transunet' else snapshot_path
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + \
        str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path+'_' + \
        str(args.max_iterations)[
            0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + \
        str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + \
        str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s' + \
        str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_' + args.loss
    snapshot_path = snapshot_path + '_mm'+str(args.momentum)
    snapshot_path = snapshot_path + '_decaylr'+str(args.decay_lr)
    snapshot_path = snapshot_path + '_lam'+str(args.lam)
    snapshot_path = snapshot_path + '_dice_ratio'+str(args.dice_ratio)
    snapshot_path = snapshot_path + '_' + args.contrastive_path[21:-4]
    snapshot_path += '_' + args.corrupt_type + str(args.corrupt_ratio)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if args.model == 'transunet':
        print("Using TransUnet!")
        # for transunet
        config = CONFIGS_ViT_seg[args.vit_name]
        config.n_classes = args.num_classes
        config.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config.patches.grid = (int(
                args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        if args.load_from == 'contrastive' and args.contrastive_path != 'default':
            config.contrastive_path = args.contrastive_path
        net = ViT_seg(config, img_size=args.img_size,
                      num_classes=config.n_classes).cuda()

    elif args.model == 'swinunet':
        print("Using SwinUnet!")
        # for swinunet
        config = get_config(args)
        net = Swin_ViT_seg(config, img_size=args.img_size,
                           num_classes=args.num_classes).cuda()
    else:
        raise ValueError("Model not available!")

    net.load_from(config, args.load_from)
    trainer = {'Synapse': trainer_synapse, }
    trainer[dataset_name](args, net, snapshot_path)
