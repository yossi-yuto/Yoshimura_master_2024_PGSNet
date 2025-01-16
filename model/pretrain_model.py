import pdb

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import torchvision.models as models
import numpy as np

from model.pgsnet import Conformer, T_Predict, Early_Fusion

class Phase1Model(nn.Module):
    def __init__(self, in_dim, new_model=False):
        super(Phase1Model, self).__init__()
        self.conformer1 = Conformer(in_chans=in_dim, patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.image_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        # self.image_cr3 = nn.Sequential(nn.Conv2d(1536, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        # self.image_cr2 = nn.Sequential(nn.Conv2d(768, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        # self.image_cr1 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.new_model = new_model
        if self.new_model:
            self.predict_c1 = nn.Conv2d(256+1, 1, 7, 1, 3)
            self.predict_t1 = T_Predict(576+1)
        else:
            self.predict_c1 = nn.Conv2d(256, 1, 7, 1, 3)
            self.predict_t1 = T_Predict(576)

    def forward(self, image: torch.Tensor, non_similar_map: torch.Tensor=None):
        image_conv_features, image_tran_features = self.conformer1(image)
        image_layer1 = image_conv_features[3]
        image_layer2 = image_conv_features[7]
        image_layer3 = image_conv_features[10]
        image_layer4 = image_conv_features[11]
        image_t4 = image_tran_features[11]
        image_cr4 = self.image_cr4(image_layer4)
        if self.new_model:
            image_cr4 = torch.cat([image_cr4, non_similar_map], dim=1)
            B, N, _ = image_t4.shape
            non_similar_map = F.interpolate(non_similar_map, size=(np.sqrt(N - 1), np.sqrt(N - 1)), mode='bilinear', align_corners=True)
            non_similar_map = non_similar_map.view(B, N, -1)
            image_t4 = torch.cat([image_t4, non_similar_map], dim=-1)

        predict_c1 = self.predict_c1(image_cr4)
        predict_t1 = self.predict_t1(image_t4)
        predict_c1 = F.interpolate(predict_c1, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t1 = F.interpolate(predict_t1, size=image.size()[2:], mode='bilinear', align_corners=True)
        return predict_c1, predict_t1


class Phase2Model(nn.Module):
    def __init__(self, in_dim, new_model=False):
        super(Phase2Model, self).__init__()
        self.new_model = new_model
        self.early_fusion_aolp = Early_Fusion()
        
        self.conformer2 = Conformer(in_chans=in_dim, patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.aolp_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        # self.aolp_cr3 = nn.Sequential(nn.Conv2d(1536, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        # self.aolp_cr2 = nn.Sequential(nn.Conv2d(768, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        # self.aolp_cr1 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.predict_c2 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t2 = T_Predict(576)

    def forward(self, aolps: torch.Tensor, non_similar_map: torch.Tensor):
        if self.new_model:
            aolp_track = aolps[:, 3:]
            aolps = aolps[:, :3]
            early_fusion_aolp = self.early_fusion_aolp(aolps)
            early_fusion_aolp = torch.cat([early_fusion_aolp, aolp_track], dim=1)
        else:
            early_fusion_aolp = self.early_fusion_aolp(aolps)
        
        aolp_conv_features, aolp_tran_features = self.conformer2(early_fusion_aolp)
        aolp_layer1 = aolp_conv_features[3]
        aolp_layer2 = aolp_conv_features[7]
        aolp_layer3 = aolp_conv_features[10]
        aolp_layer4 = aolp_conv_features[11]
        aolp_t4 = aolp_tran_features[11]
        aolp_cr4 = self.aolp_cr4(aolp_layer4)
        predict_c2 = self.predict_c2(aolp_cr4)
        predict_t2 = self.predict_t2(aolp_t4)
        predict_c2 = F.interpolate(predict_c2, size=early_fusion_aolp.size()[2:], mode='bilinear', align_corners=True)
        predict_t2 = F.interpolate(predict_t2, size=early_fusion_aolp.size()[2:], mode='bilinear', align_corners=True)
        return predict_c2, predict_t2


class Phase3Model(nn.Module):
    def __init__(self, in_dim, new_model=False):
        super(Phase3Model, self).__init__()
        self.new_model = new_model
        self.early_fusion_dolp = Early_Fusion()
        self.conformer3 = Conformer(in_chans=in_dim, patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.dolp_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        # self.dolp_cr3 = nn.Sequential(nn.Conv2d(1536, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        # self.dolp_cr2 = nn.Sequential(nn.Conv2d(768, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        # self.dolp_cr1 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.predict_c3 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t3 = T_Predict(576)

    def forward(self, dolps: torch.Tensor):
        if self.new_model:
            dolp_track = dolps[:, 3:]
            dolps = dolps[:, :3]
            early_fusion_dolp = self.early_fusion_dolp(dolps)
            early_fusion_dolp = torch.cat([early_fusion_dolp, dolp_track], dim=1)
        else:
            early_fusion_dolp = self.early_fusion_dolp(dolps)
        
        dolp_conv_features, dolp_tran_features = self.conformer3(early_fusion_dolp)
        dolp_layer1 = dolp_conv_features[3]
        dolp_layer2 = dolp_conv_features[7]
        dolp_layer3 = dolp_conv_features[10]
        dolp_layer4 = dolp_conv_features[11]
        dolp_t4 = dolp_tran_features[11]
        dolp_cr4 = self.dolp_cr4(dolp_layer4)
        predict_c3 = self.predict_c3(dolp_cr4)
        predict_t3 = self.predict_t3(dolp_t4)
        predict_c3 = F.interpolate(predict_c3, size=early_fusion_dolp.size()[2:], mode='bilinear', align_corners=True)
        predict_t3 = F.interpolate(predict_t3, size=early_fusion_dolp.size()[2:], mode='bilinear', align_corners=True)
        return predict_c3, predict_t3


class VideoSegNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conformer = Conformer(in_chans=in_dim, patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.image_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        # self.image_cr3 = nn.Sequential(nn.Conv2d(1536, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        # self.image_cr2 = nn.Sequential(nn.Conv2d(768, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        # self.image_cr1 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.predict_c1 = nn.Conv2d(256, 66, 7, 1, 3)
        self.predict_t1 = T_Predict(576, 66)

    def forward(self, image: torch.Tensor):
        image_conv_features, image_tran_features = self.conformer(image)
        image_layer1 = image_conv_features[3]
        image_layer2 = image_conv_features[7]
        image_layer3 = image_conv_features[10]
        image_layer4 = image_conv_features[11]
        image_t4 = image_tran_features[11]
        image_cr4 = self.image_cr4(image_layer4)
        predict_c1 = self.predict_c1(image_cr4)
        predict_t1 = self.predict_t1(image_t4)
        predict_c1 = F.interpolate(predict_c1, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t1 = F.interpolate(predict_t1, size=image.size()[2:], mode='bilinear', align_corners=True)
        return predict_c1, predict_t1