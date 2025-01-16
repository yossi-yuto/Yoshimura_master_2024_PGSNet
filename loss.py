import numpy
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(inputs.size(0), -1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = torch.sum(inputs * targets, dim=-1)
        total = torch.sum(inputs + targets, dim=-1)
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
        return torch.mean(1 - IoU)
    
    
class DICE_BCE_losses(nn.Module):
    def __init__(self, final_weight=2):
        super().__init__()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = DiceLoss(mode="binary")
        self.w_final = final_weight
        
    def forward(self, preds: tuple, target_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        for i, pred in enumerate(preds):
            if i == len(preds) - 1:
                sum_loss += self.combined_loss_fn(pred, target_mask) * self.w_final
            else:
                sum_loss += self.combined_loss_fn(pred, target_mask)
        return sum_loss
    
    def combined_loss_fn(self, pred, target):
        return self.bce_loss_fn(pred, target) + self.dice_loss_fn(pred, target)


class DiceFocal_losses(nn.Module):
    def __init__(self, final_weight=2):
        super().__init__()
        self.focal_loss_fn = FocalLoss(mode="binary")
        self.dice_loss_fn = DiceLoss(mode="binary")
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.w_final = final_weight

    def forward(self, preds: tuple, target_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, pred in enumerate(preds):
            if i == len(preds) - 1:
                # print("final loss")
                focal_loss = self.focal_loss_fn(pred, target_mask)
                dice_loss = self.dice_loss_fn(pred, target_mask)
                edge_loss = self.bce_loss_fn(pred, edge_mask)
                # print(f" Focal loss: {focal_loss}")
                # print(f" Dice loss: {dice_loss}")
                # print(f" Edge loss: {edge_loss}")
                sum_loss = focal_loss + dice_loss * self.w_final + edge_loss
                sum_loss = sum_loss * self.w_final  
                losses.append(sum_loss)
            else:
                # print(f"Prediction {i}")
                focal_loss = self.focal_loss_fn(pred, target_mask)
                dice_loss = self.dice_loss_fn(pred, target_mask)
                # print(f" Focal loss: {focal_loss}")
                # print(f" Dice loss: {dice_loss}")
                sum_loss = focal_loss + dice_loss
                losses.append(sum_loss)
        return losses
    
    def combined_loss_fn(self, pred, target):
        return self.focal_loss_fn(pred, target) + self.dice_loss_fn(pred, target)


class DiceFocalSoftloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss_fn = FocalLoss(mode="multiclass")
        self.dice_loss_fn = DiceLoss(mode="multiclass")
        
    def forward(self, pred: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for p in pred:
            losses.append(self.combined_loss_fn(p, target_mask))
        return losses

    def combined_loss_fn(self, pred, target_mask):
        target_mask = torch.argmax(target_mask.long(), dim=1)
        return self.focal_loss_fn(pred, target_mask) + self.dice_loss_fn(pred, target_mask)
        

        
    