import copy
import logging
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from scipy import ndimage
# TransUnet
from networks.vit_seg_modeling import Transformer
# SwinUnet
from networks.swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed, BasicLayer, PatchMerging

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class TransUnetEncoder(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        super(TransUnetEncoder, self).__init__()
        self.num_classes = config.num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, config.img_size, vis)
        self.config = config
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        return x

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class SwinUnetEncoder(nn.Module):
    def __init__(self, config, zero_head=False, vis=False, norm_layer=nn.LayerNorm):
        super(SwinUnetEncoder, self).__init__()
        self.num_classes = config.num_classes
        self.zero_head = zero_head

        self.num_layers = len(config.DEPTHS)
        self.embed_dim = config.EMBED_DIM
        self.ape = config.APE
        self.patch_norm = config.PATCH_NORM
        self.num_features = int(config.EMBED_DIM * 2 ** (len(config.DEPTHS) - 1))
        self.num_features_up = int(config.EMBED_DIM * 2)
        self.mlp_ratio = config.MLP_RATIO
        self.in_chans = config.IN_CHANS

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(  img_size=config.img_size, 
                                        patch_size=config.PATCH_SIZE, 
                                        in_chans=config.IN_CHANS, 
                                        embed_dim=config.EMBED_DIM,
                                        norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.EMBED_DIM))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=config.DROP_RATE)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.DROP_PATH_RATE, sum(config.DEPTHS))]  

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(config.EMBED_DIM * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=config.DEPTHS[i_layer],
                               num_heads=config.NUM_HEADS[i_layer],
                               window_size=config.WINDOW_SIZE,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=config.QKV_BIAS, 
                               qk_scale=config.QK_SCALE,
                               drop=config.DROP_RATE, 
                               attn_drop=config.ATTN_DROP_RATE,
                               drop_path=dpr[sum(config.DEPTHS[:i_layer]):sum(config.DEPTHS[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=config.use_checkpoint)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,self.in_chans,1,1)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        return x

    def load_from(self, config):
        pretrained_path = config.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            # for upsampling only
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            # for EncoderModel, prepending 'model.'
            full_dict = {'model.'+k:v for k, v in pretrained_dict.items()}
            # filter out mismatching sizes
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class EncoderModel(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1

    * arch = {'transunet','swinunet'}
    """
    def __init__(self, config, zero_head=False, vis=False):
        super(EncoderModel, self).__init__()

        if config.arch=='transunet':
            self.model = TransUnetEncoder(config, zero_head=zero_head, vis=vis) # (B,C,F)
            self.grid_H = config.patches.grid[0]
            self.grid_W = config.patches.grid[1]
            self.encoder_feature_dim = self.model.transformer.encoder.encoder_norm.bias.size(dim=0) 
        elif config.arch=='swinunet':
            self.model = SwinUnetEncoder(config, zero_head=zero_head, vis=vis)
            self.grid_H = int(config.img_size/2**(len(config.DEPTHS)-1)/config.PATCH_SIZE)
            self.grid_W = int(config.img_size/2**(len(config.DEPTHS)-1)/config.PATCH_SIZE)
            self.encoder_feature_dim = self.model.norm.bias.size(dim=0)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self.encoder_feature_dim, config.feature_dim)


    def forward(self, x):
        x = self.model(x) # (B,C,F)
        B,C,F = x.shape
        assert C==self.grid_H*self.grid_W, "Grid size mismatch: ({:d}, {:d}), {:d}".format(self.grid_H, self.grid_W, C)
        x = x.permute((0,2,1)).view(B, F, self.grid_H, self.grid_H) # (B,F,H,W)
        x = self.pooling(x) # (B,F,1,1)
        x = self.flatten(x) # (B,F)
        x = self.fc(x)
        # note: not normalized here
        return x

    def load_from(self, config):
        if config.arch=='transunet':
            weights=np.load(config.pretrained_path)
            self.model.load_from(weights)
        elif config.arch=='swinunet':
            self.model.load_from(config)
        else: 
            print("none pretrain")

class MoCo(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        '''
        * dim =         dim of encoder output (feature)
        * K =           key dictionary size
        * m =           momentum for key encoder
        * T =           temperature for infoNCE
        * arch =        encoder backbone architecture
        * symmetric =   using symmetric / asymmetric loss
        '''
        super(MoCo, self).__init__()

        self.K = config.moco_k
        self.m = config.moco_m
        self.T = config.moco_t
        self.symmetric = config.symmetric

        # create the encoders
        self.encoder_q = EncoderModel(config, zero_head=zero_head, vis=vis)
        self.encoder_k = EncoderModel(config, zero_head=zero_head, vis=vis)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(config.feature_dim, config.moco_k))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss

    def load_from(self, config):
        self.encoder_q.load_from(config)
        self.encoder_k.load_from(config)


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, config, zero_head=False, vis=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        self.encoder = EncoderModel(config, zero_head=zero_head, vis=vis)

        # build a 3-layer projector: self.encoder.encoder_feature_dim -> config.feature_dim
        F = self.encoder.fc.weight.shape[1] #self.encoder.encoder_feature_dim = config.feature_dim
        self.encoder.fc = nn.Sequential(nn.Linear(F, F, bias=False),
                                        nn.BatchNorm1d(F),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(F, F, bias=False),
                                        nn.BatchNorm1d(F),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(config.feature_dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential( nn.Linear(config.feature_dim, config.pred_dim, bias=False),
                                        nn.BatchNorm1d(config.pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(config.pred_dim, config.feature_dim)) # output layer
        
        # define loss function (criterion) and optimizer
        self.criterion = nn.CosineSimilarity(dim=1)


    def encode_pred(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxF
        z2 = self.encoder(x2) # NxF

        p1 = self.predictor(z1) # NxF
        p2 = self.predictor(z2) # NxF

        return p1, p2, z1.detach(), z2.detach()

    def forward(self, x1, x2):
        # compute output and loss
        p1, p2, z1, z2 = self.encode_pred(x1, x2)
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss

    def load_from(self, config):
        self.encoder.load_from(config)


class BYOL(nn.Module):
    """
    BYOL model
    """
    def __init__(self, config, zero_head=False, vis=False):
        super(BYOL, self).__init__()
        self.m = config.momentum_encoder
        self.config = config

        # initialize online and target encoders
        self.encoder_online = EncoderModel(config, zero_head=zero_head, vis=vis)
        self.encoder_target = EncoderModel(config, zero_head=zero_head, vis=vis)
        
        # append 3-layer MLP to both online and target encoders
        self._append_mlp(self.encoder_online)
        self._append_mlp(self.encoder_target)

        # build a 2-layer predictor
        self.predictor = nn.Sequential( nn.Linear(config.feature_dim, config.pred_dim, bias=False),
                                        nn.BatchNorm1d(config.pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(config.pred_dim, config.feature_dim)) # output layer
        
        # define loss function (criterion) and optimizer (note: equivalent to l_2 loss with normalized features)
        self.criterion = nn.CosineSimilarity(dim=1)


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        # momentum update target encoder
        with torch.no_grad():  
            self._momentum_update_target_encoder()

        loss = -(self._loss_asymmetric(x1,x2) + self._loss_asymmetric(x2,x1)) * 0.5
        return loss


    def load_from(self, config):
        # load pretrained online encoder
        self.encoder_online.load_from(config)
        # copy online encoder to target encoder
        for param_online, param_target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False  # not update by gradient


    def _loss_asymmetric(self, x1, x2):
        # compute features for one view
        z1 = self.encoder_online(x1) # NxF
        p1 = self.predictor(z1) # NxF
        z2 = self.encoder_target(x2).detach() # NxF
        loss = -self.criterion(p1, z2).mean()
        return loss
    
    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of target encoder
        """
        for param_online, param_target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            param_target.data = param_target.data * self.m + param_online.data * (1. - self.m)


    def _append_mlp(self, net):
        """
        Replace 1-layer fc with 3-layer projector, in-place, as in SimSiam: 
            net.encoder_feature_dim -> config.feature_dim
        """
        if not hasattr(net, 'fc'):
            return net
        F = net.fc.weight.shape[1] # net.encoder_feature_dim = config.feature_dim
        net.fc = nn.Sequential( nn.Linear(F, F, bias=False),
                                nn.BatchNorm1d(F),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(F, F, bias=False),
                                nn.BatchNorm1d(F),
                                nn.ReLU(inplace=True), # second layer
                                net.fc,
                                nn.BatchNorm1d(self.config.feature_dim, affine=False)) # output layer
        net.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
