import argparse
# import logging
import math
import pickle
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
from contrastive.configs import get_basic_config, get_transunet_config, get_swinunet_config, get_synapse_config
from contrastive.datasets import get_synapse_pair_loader
from contrastive.models import BYOL, MoCo, SimSiam

def pretrain():
    parser = argparse.ArgumentParser(description='Segmentation: Dataset=Synapse, Encoder backbone=SwinUnet')
    parser.add_argument('-a', '--arch', default='transunet', help='encoder backbones: transunet / swinunet')
    parser.add_argument('--aug-strength', default=8, type=int, help='data augmentation strength')
    parser.add_argument('--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--contrastive', default='simsiam', help='contrastive models: byol / simsiam')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--dataset', type=str, default='Synapse', help='datasets: Synapse')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--feature-dim', default=768, type=int, help='feature dimension')
    parser.add_argument('--img-size', default=224, type=int, help='Input image size to model (independent of raw image size)')
    parser.add_argument('--lr', '--learning-rate', default=0.2, type=float, metavar='LR', 
                        help='baseline initial learning rate (lr_init = lr*batch_size/256)', dest='lr')
    parser.add_argument('--momentum-encoder', default=0.99, type=float, help='momentum of updating target encoder')
    parser.add_argument('--num-gpu', default=1, type=int, help='number of GPU. Check with nvidia-smi')
    parser.add_argument('--num-workers', default=8, type=int, help='num_worksd for dataloader')
    parser.add_argument('--pred-dim', default=256, type=int, help='simsiam prediction dimension') # for simsiam
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, 
                        help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    # for moco
    parser.add_argument('--moco-k', default=768, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')


    arch_config = { 'transunet':    get_transunet_config, 
                    'swinunet':     get_swinunet_config}
    data_config = { 'Synapse':      get_synapse_config}
    
    argin = parser.parse_args()
    config = get_basic_config(argin)
    config = data_config[config.dataset](config)
    config = arch_config[config.arch](config)
    
    # dataloader
    pretrain_loader = get_synapse_pair_loader(config, aug_strength=config.aug_strength)
    
    # model
    contrastive_model = {'byol' : BYOL, 'moco' : MoCo, 'simsiam' : SimSiam}
    model = contrastive_model[config.contrastive](config=config)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    
    # load pretrained / resume model
    epoch_start = 1
    if config.resume != '':
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(config.resume))
    else:
        model.load_from(config=config)

    # model parallel
    if config.num_gpu > 1:
        model = nn.DataParallel(model)
    else:
        model = model.cuda()

    # logging
    results = { 'train_loss': [], 
                # 'test_acc@1': [],
                }
    if not os.path.exists(config.results_dir):
        os.mkdir(config.results_dir)
    # dump config
    with open(config.results_dir + '/config.pickle', 'wb') as fid:
        pickle.dump(config.__dict__, fid)

    # training loop
    for epoch in range(epoch_start, config.epochs + 1):
        train_loss = train_epoch(model, pretrain_loader, optimizer, epoch, config)
        results['train_loss'].append(train_loss)
        # test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, config)
        # results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(config.results_dir + '/log.csv', index_label='epoch')
        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, config.results_dir + '/model_last.pth')
    
    return "Pretraining Finished!"

# train for one epoch
def train_epoch(model, data_loader, optimizer, epoch, config):
    model.train()
    adjust_learning_rate(optimizer, epoch-1, config)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(), im_2.cuda()

        loss = model(im_1, im_2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(
                epoch, config.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))
    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.lr
    if config.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config.epochs))
    else:  # stepwise lr schedule
        for milestone in config.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__=='__main__':
    pretrain()