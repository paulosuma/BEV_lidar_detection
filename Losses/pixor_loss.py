import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Pixor_loss(nn.Module):
    def __init__(self, gamma=5, beta=1.0, alpha=0.9):
        super(Pixor_loss, self).__init__()
        self.gamma = gamma
        self.beta = beta #for L1 loss
        self.alpha = alpha
        self.smoothL1loss = nn.SmoothL1Loss(beta=self.beta)

    def get_class_loss(self, prediction, mask):
        eps = 1e-10  # Small epsilon to avoid log(0)
        #Has object
        object_CE_loss = - torch.log(prediction[mask] + eps)
        scaled_object_loss = self.alpha*(1 - prediction[mask])**self.gamma*object_CE_loss

        #No object
        no_object_CE_loss = - torch.log(1 - prediction[torch.logical_not(mask)] + eps)
        scaled_no_object_loss = (1 - self.alpha)*prediction[torch.logical_not(mask)]**self.gamma*no_object_CE_loss

        return torch.mean(scaled_object_loss) + torch.mean(scaled_no_object_loss)
        
    def get_reg_loss(self, prediction, target):
        return self.smoothL1loss(prediction, target.detach())

    def forward(self, prediction, target):
        mask = target[:, :, :, 0] == 1.0

        if mask.any():
            cls_pred = prediction[:, :, :, 0]
            reg_pred, reg_target = prediction[:, :, :, 1:], target[:, :, :, 1:]

            cls_loss = self.get_class_loss(cls_pred, mask)
            reg_loss = self.get_reg_loss(reg_pred[mask], reg_target[mask])

            total_loss = cls_loss + reg_loss
            return total_loss , cls_loss, reg_loss
        else:
            #No object
            eps = 1e-7
            cls_pred = prediction[:, :, :, 0]
            no_object_CE_loss = - torch.log(1 - cls_pred + eps)
            cls_loss = torch.mean((1 - self.alpha)*cls_pred**self.gamma*no_object_CE_loss)
            zero_tensor = torch.tensor(0.0, requires_grad=True).to(prediction.device)
            print("!!", cls_loss.item())
            return cls_loss, cls_loss, zero_tensor
