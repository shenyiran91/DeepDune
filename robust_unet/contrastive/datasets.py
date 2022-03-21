# code in this file is adpated from:
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
# https://github.com/Beckschen/TransUNet/blob/main/datasets/dataset_synapse.py
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from contrastive.randaug import *
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.ndimage.interpolation import zoom


def get_synapse_pair_loader(config, aug_strength=10):
    base_dir, list_dir = config.synapse_base_dir, config.synapse_list_dir
    split='train'
    
    transform = RandomAug_Synapse(output_size=[config.img_size, config.img_size], n=aug_strength)
    dataset = Synapse_pair_unsup(base_dir, list_dir, split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    return dataloader

def get_synapse_dataset(config, split='train', pair=True, aug_strength=10):
    '''
    * split = {'train','test_vol'}
    '''
    base_dir, list_dir = config.synapse_base_dir, config.synapse_list_dir
    
    # dataset
    if pair: # augmented pair without label
        dataset = Synapse_pair_unsup(base_dir, list_dir, split, 
                    transform=RandomAug_Synapse(output_size=[config.img_size, config.img_size], n=aug_strength))
    else: # supervised image-label pair
        dataset = Synapse(base_dir, list_dir, split, transform=None)
    
    return dataset


class ToTensor_Synapse(object):
    def __init__(self, output_size=[224,224]):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) #(1,H,W)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class RandomAug_Synapse(ToTensor_Synapse):
    def __init__(self, output_size=[224, 224], n=10, m=10):
        super().__init__(output_size)
        assert n >= 1 # augmentation strength: number of augmentations applied
        assert 1 <= m <= PARAMETER_MAX # augmentation parameter
        self.n = n
        self.m = m
        self.augment_pool = medical_image_augment_pool()

    def __call__(self, sample):
        sample = super().__call__(sample)
        image = sample['image'] # torch.Tensor
        # label = sample['label']
        
        # Tensor to PILImage
        image = transforms.ToPILImage()(image)
        # label_min, label_max = label.min(), label.max()
        # label_scaled = (label-label_min)/(label_max-label_min)
        # label = transforms.ToPILImage()(label_scaled.float())
        
        # RandAugment
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias, modify_label in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                image = op(image, v=v, max_v=max_v, bias=bias)
        
        # PILImage to Tensor
        image = transforms.ToTensor()(image)
        # label = transforms.ToTensor()(label)
        # label = label*(label_max-label_min)+label_min

        sample = {'image': image}
        return sample


# Basic dataset, supervised
class Synapse(Synapse_dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        if transform is None:
            transform = ToTensor_Synapse()
        super().__init__(base_dir, list_dir, split, transform=transform)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)

# Two-view dataset, unsupervised
class Synapse_pair_unsup(Synapse):
    def __getitem__(self, idx):
        sample_1 = super().__getitem__(idx)
        sample_2 = super().__getitem__(idx)
        return sample_1['image'], sample_2['image']

