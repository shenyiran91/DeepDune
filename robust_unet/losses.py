import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import math

# not good for rotation and flipping!
class AdaWeightLoss(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224, lam: float = 0.2):
        super(AdaWeightLoss, self).__init__()
        self.lam = lam
        self.acc_loss_array = torch.zeros((num_examp, img_size, img_size)).cuda()
        self.img_size = img_size

    def forward(self, output, label, index):
        tmp_loss = F.cross_entropy(output, label, reduction='none')
        self.acc_loss_array[index] += tmp_loss.detach()
        rate = self.lam + (1 - self.lam) * self.acc_loss_array[index]
        # rate = 1.0
        total = self.img_size * self.img_size * len(output)
        return 1.0 / total * torch.sum(tmp_loss / rate)


class AdaImageLoss(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224, lam: float = 0.2):
        super(AdaImageLoss, self).__init__()
        self.lam = lam
        self.acc_loss_array = torch.zeros(num_examp).cuda()
        self.img_size = img_size

    def forward(self, output, label, index):
        tmp_loss = F.cross_entropy(output, label, reduction='none')
        self.acc_loss_array[index] += tmp_loss.mean((1, 2)).detach()
        rate = self.lam + (1 - self.lam) * self.acc_loss_array[index]
        total = self.img_size * self.img_size * len(output)
        return 1.0 / total * torch.sum(tmp_loss / rate[:, None, None])


class AdaSingleLoss(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224, lam: float = 0.2):
        super(AdaSingleLoss, self).__init__()
        self.lam = lam
        self.acc_loss = 0

    def forward(self, output, label, index):
        tmp_loss = F.cross_entropy(output, label)
        self.acc_loss += tmp_loss.detach()
        rate = self.lam + (1 - self.lam) * self.acc_loss
        return tmp_loss / rate


class CELoss(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224, lam: float = 0.2):
        super(CELoss, self).__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, output, label, index):
        return self.ce_loss(output, label)


class Losses:
    ce_loss = CELoss
    ada_weight = AdaWeightLoss
    ada_image = AdaImageLoss
    ada_single = AdaSingleLoss
        