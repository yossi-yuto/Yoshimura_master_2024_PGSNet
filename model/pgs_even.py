import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .proposed import Conformer, Early_Fusion, BasicConv, T_Predict, MSDP, Fusion_T, Fusion_C, Decoder, Global_Attention

logger = logging.getLogger(__name__)
logger.info('PGSNetEven is imported.')


class PGSNetEven(nn.Module):
    def __init__(self):
        super(PGSNetEven, self).__init__()

        # Encoders
        self.conformer1 = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.early_fusion_aolp = Early_Fusion()
        self.conformer2 = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.early_fusion_dolp = Early_Fusion()
        self.conformer3 = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)

        self.image_cr4 = BasicConv(1536, 256, 3, 1, 1)
        self.aolp_cr4 = BasicConv(1536, 256, 3, 1, 1)
        self.dolp_cr4 = BasicConv(1536, 256, 3, 1, 1)

        self.predict_c_rgb = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t_rgb = T_Predict(576)
        self.predict_c_aolp = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t_aolp = T_Predict(576)
        self.predict_c_dolp = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t_dolp = T_Predict(576)

        self.image_cr3 = BasicConv(1536, 1536 // 3, 3, 1, 1)
        self.image_cr2 = BasicConv(768, 768 // 3, 3, 1, 1)
        self.image_cr1 = BasicConv(384, 384 // 3, 3, 1, 1)
        self.aolp_cr3 = BasicConv(1536, 1536 // 3, 3, 1, 1)
        self.aolp_cr2 = BasicConv(768, 768 // 3, 3, 1, 1)
        self.aolp_cr1 = BasicConv(384, 384 // 3, 3, 1, 1)
        self.dolp_cr3 = BasicConv(1536, 1536 // 3, 3, 1, 1)
        self.dolp_cr2 = BasicConv(768, 768 // 3, 3, 1, 1)
        self.dolp_cr1 = BasicConv(384, 384 // 3, 3, 1, 1)

        self.fusion_cr3 = BasicConv(1536, 128, 3, 1, 1)
        self.fusion_cr2 = BasicConv(768, 64, 3, 1, 1)
        self.fusion_cr1 = BasicConv(384, 32, 3, 1, 1)

        self.predict_m4 = nn.Conv2d(256 * 3, 1, 7, 1, 3)
        self.predict_m3 = nn.Conv2d(128, 1, 7, 1, 3)
        self.predict_m2 = nn.Conv2d(64, 1, 7, 1, 3)
        self.predict_m1 = nn.Conv2d(32, 1, 7, 1, 3)

        self.msdp = MSDP(256)
        self.fusion_t = Fusion_T(576)
        self.fusion_c = Fusion_C(13, 576)

        self.decoder43 = Decoder(256)
        self.decoder32 = Decoder(128)
        self.decoder21 = Decoder(64)

        self.ga4 = Global_Attention(256, 576)
        self.ga3 = Global_Attention(128, 576)
        self.ga2 = Global_Attention(64, 576)
        self.ga1 = Global_Attention(32, 576)

        self.predict_t = T_Predict(576)
        self.predict4 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict3 = nn.Conv2d(128, 1, 7, 1, 3)
        self.predict2 = nn.Conv2d(64, 1, 7, 1, 3)
        self.predict1 = nn.Conv2d(32, 1, 7, 1, 3)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, image: torch.Tensor, aolps: list, dolps: list):
        image_conv_features, image_tran_features = self.conformer1(image)
        image_layer1 = image_conv_features[3]
        image_layer2 = image_conv_features[7]
        image_layer3 = image_conv_features[10]
        image_layer4 = image_conv_features[11]
        image_t4 = image_tran_features[11]

        early_fusion_aolp = self.early_fusion_aolp(aolps)
        aolp_conv_features, aolp_tran_features = self.conformer2(early_fusion_aolp)
        aolp_layer1 = aolp_conv_features[3]
        aolp_layer2 = aolp_conv_features[7]
        aolp_layer3 = aolp_conv_features[10]
        aolp_layer4 = aolp_conv_features[11]
        aolp_t4 = aolp_tran_features[11]

        early_fusion_dolp = self.early_fusion_dolp(dolps)
        dolp_conv_features, dolp_tran_features = self.conformer3(early_fusion_dolp)
        dolp_layer1 = dolp_conv_features[3]
        dolp_layer2 = dolp_conv_features[7]
        dolp_layer3 = dolp_conv_features[10]
        dolp_layer4 = dolp_conv_features[11]
        dolp_t4 = dolp_tran_features[11]

        image_cr4 = self.image_cr4(image_layer4)
        predict_c_rgb = self.predict_c_rgb(image_cr4)
        predict_t_rgb = self.predict_t_rgb(image_t4)
        predict_c_rgb = F.interpolate(predict_c_rgb, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t_rgb = F.interpolate(predict_t_rgb, size=image.size()[2:], mode='bilinear', align_corners=True)

        aolp_cr4 = self.aolp_cr4(aolp_layer4)
        predict_c_aolp = self.predict_c_aolp(aolp_cr4)
        predict_t_aolp = self.predict_t_aolp(aolp_t4)
        predict_c_aolp = F.interpolate(predict_c_aolp, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t_aolp = F.interpolate(predict_t_aolp, size=image.size()[2:], mode='bilinear', align_corners=True)

        dolp_cr4 = self.dolp_cr4(dolp_layer4)
        predict_c_dolp = self.predict_c_dolp(dolp_cr4)
        predict_t_dolp = self.predict_t_dolp(dolp_t4)
        predict_c_dolp = F.interpolate(predict_c_dolp, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t_dolp = F.interpolate(predict_t_dolp, size=image.size()[2:], mode='bilinear', align_corners=True)

        fusion_t = self.fusion_t(image_t4, aolp_t4, dolp_t4)   # GCG 
        fusion_c = self.fusion_c(image_cr4, aolp_cr4, dolp_cr4, image_t4, aolp_t4, dolp_t4) #
        msdp = self.msdp(fusion_c)

        ga4 = self.ga4(msdp, fusion_t) # AE4
        feat_map3 = self.fusion_cr3(torch.cat([self.image_cr3(image_layer3), self.aolp_cr3(aolp_layer3), self.dolp_cr3(dolp_layer3)], 1))
        decoder43 = self.decoder43(ga4, feat_map3)
        ga3 = self.ga3(decoder43, fusion_t) # AE3

        feat_map2 = self.fusion_cr2(torch.cat([self.image_cr2(image_layer2), self.aolp_cr2(aolp_layer2), self.dolp_cr2(dolp_layer2)], 1))
        decoder32 = self.decoder32(ga3, feat_map2)
        ga2 = self.ga2(decoder32, fusion_t) # AE2

        feat_map1 = self.fusion_cr1(torch.cat([self.image_cr1(image_layer1), self.aolp_cr1(aolp_layer1), self.dolp_cr1(dolp_layer1)], 1))
        decoder21 = self.decoder21(ga2, feat_map1)
        ga1 = self.ga1(decoder21, fusion_t) # AE1

        predict_t = self.predict_t(fusion_t)
        predict4 = self.predict4(ga4)
        predict3 = self.predict3(ga3)
        predict2 = self.predict2(ga2)
        predict1 = self.predict1(ga1)

        predict_m4 = self.predict_m4(torch.cat([image_cr4, aolp_cr4, dolp_cr4], 1))
        predict_m3 = self.predict_m3(feat_map3)
        predict_m2 = self.predict_m2(feat_map2)
        predict_m1 = self.predict_m1(feat_map1)

        predict_t = F.interpolate(predict_t, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_m4 = F.interpolate(predict_m4, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_m3 = F.interpolate(predict_m3, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_m2 = F.interpolate(predict_m2, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_m1 = F.interpolate(predict_m1, size=image.size()[2:], mode='bilinear', align_corners=True)

        # outputs = (predict_c_rgb, predict_t_rgb, predict_c_aolp, predict_t_aolp, predict_c_dolp, predict_t_dolp, predict_m4, predict_m3, predict_m2, predict_m1,
        #            predict_t, predict4, predict3, predict2, predict1)
        outputs = (predict_m4, predict_m3, predict_m2, predict_m1, predict_t, predict4, predict3, predict2, predict1,)

        return outputs
