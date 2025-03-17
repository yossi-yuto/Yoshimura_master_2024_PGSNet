import os
import math
import pdb
from datetime import datetime
from argparse import ArgumentParser
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import BinaryFBetaScore
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

from loss import DICE_BCE_losses, DiceFocal_losses
from earlystop import EarlyStopping
from preprocessing import PreProcessing
from config import set_seed, parse_args, create_mirror_dataset
from metrics import get_maxFscore_and_threshold
from plot import save_plots, plot_and_save
from layer_freeze import freeze_out_layer
from model.proposed_net import Network


def main():
    
    args = parse_args()
    # 乱数シード固定
    set_seed(args.random_seed)
    
    # データセットの作成
    train_loader, val_loader, _ = create_mirror_dataset(args)
    
    # 結果保存用ディレクトリの作成
    dir_checkpoint = os.path.join(args.result_dir, "ckpt")
    dir_loss_metrics_graph = os.path.join(args.result_dir, "eval")
    dir_val_outputs = os.path.join(args.result_dir,  "check_outputs")
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(dir_loss_metrics_graph, exist_ok=True)
    os.makedirs(dir_val_outputs, exist_ok=True)
    
    # model instance
    netfile = importlib.import_module("model." + args.model)
    model = netfile.Network(in_dim=3).cuda()
    
    # preprocessing setting
    print("Preprocessing setting...")
    preprocessing = PreProcessing(grid_size=args.grid_size, relative_flow=True)

    # loss function
    dice_loss_fn = DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    # weight init
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.Conv2d):
    #         torch.nn.init.kaiming_normal_(layer.weight)
    #     elif isinstance(layer, torch.nn.Linear):
    #         torch.nn.init.xavier_normal_(layer.weight)
    
    # network layer freeze
    if args.phase == 1:
        # pgs_ckpt_path = os.path.join(dir_checkpoint.replace("/proposed", "/PGS"), "best_weight.pth")
        pgs_ckpt_path = "/data2/yoshimura/mirror_detection/PGSNet/work20250101/results/20250121_seed42/PGS/ash/ckpt/best_weight.pth"
        print("Loading PGSNet pretrained model...", pgs_ckpt_path)
        model.load_state_dict(torch.load(pgs_ckpt_path, weights_only=True), strict=False)
        for name, param in model.named_parameters():
            if 'cross_attention_module' in name:
                print("Training layer: ", name)
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        def compute_loss_phase(output_dict: dict, tgt_mask_torch: torch.Tensor, epoch:int, final_weight: float = 1.0, verbose: bool = False) -> list:
            print("This loss function is for phase 1.")
            sum_loss = 0
            tgt_mask_cuda = tgt_mask_torch.cuda()
            for key, value in output_dict.items():
                if 'opflow' in key:
                    loss = bce_loss_fn(value, tgt_mask_cuda.to(torch.float32)) + dice_loss_fn(value, tgt_mask_cuda)
                    sum_loss += loss
                    if verbose:
                        print(f"{key} loss: {loss.item()}")
                else:
                    pass
            return sum_loss
        
        compute_loss = compute_loss_phase

    elif args.phase == 2:
        """ conformer 事前学習済み """
        imagenet_pretraind_conformer = "/data2/yoshimura/mirror_detection/PGSNet/pretrain_param/Conformer_base_patch16.pth"
        print("Loading Conformer pretrained model...", imagenet_pretraind_conformer)
        model.conformer1.load_state_dict(torch.load(imagenet_pretraind_conformer, weights_only = True))
        model.conformer2.load_state_dict(torch.load(imagenet_pretraind_conformer, weights_only = True))
        model.conformer3.load_state_dict(torch.load(imagenet_pretraind_conformer, weights_only = True))
        for name, param in model.named_parameters():
            if 'conformer' in name:
                param.requires_grad = False
            else: 
                param.requires_grad = True
        
        # pgs_ckpt_path = os.path.join(dir_checkpoint.replace("/proposed", "/PGS"), "best_weight.pth")
        # print("Loading PGSNet pretrained model...", pgs_ckpt_path)
        # model.load_state_dict(torch.load(pgs_ckpt_path, weights_only=True), strict=False)
        
        # pretrained_modelからcross_attention_moduleのみのパラメータを読み込む
        # pretrained_ckpt_path = os.path.join(dir_checkpoint, "best_weight.pth")
        # cross_attention_module_dict = torch.load(pretrained_ckpt_path, weights_only=True)
        # for name, param in model.named_parameters():
        #     if 'cross_attention_module' in name:
        #         print("Loading layer: ", name)
        #         param.data = cross_attention_module_dict[name].data
        
        def compute_loss_phase2(output_dict: dict, tgt_mask_torch: torch.Tensor, epoch: int, final_weight: float = 1.0, verbose: bool = False, ) -> list:
            print("This loss function is for phase 2.")
            sum_loss = 0
            tgt_mask_cuda = tgt_mask_torch.cuda()  # 1回だけ CUDA に送る
            for key, value in output_dict.items():
                loss = bce_loss_fn(value, tgt_mask_cuda.to(torch.float32)) + dice_loss_fn(value, tgt_mask_cuda)
                sum_loss += loss
                if verbose:
                    print("layer: ", key, "loss: ", loss.item())
            print(f"total loss: {sum_loss.item()}")
            return sum_loss
        
        compute_loss = compute_loss_phase2
        
        # パラメータの凍結
        for name, param in model.named_parameters():
            if 'conformer' in name:
                param.requires_grad = False
            
    else:
        print("Training all layers.")
    
    
    # metrics
    metrics_fn = BinaryFBetaScore(beta=0.5)
    
    # early stopping
    es = EarlyStopping(verbose=True, 
                        patience=args.patient, 
                        filepath=os.path.join(dir_checkpoint, "best_weight.pth"))
    
    # optimizer
    # optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    optimizer = optim.AdamW(model.parameters())
    
    # recoder list 
    train_loss_epochs = []
    val_loss_epochs = []
    train_score_epochs = []
    val_score_epochs = []
    
    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch+1))
        
        ''' training '''
        model.train()
        train_loss_iter = []
        train_score_iter = []
        for tgt_image_torch, tgt_aolps_torch, tgt_dolps_torch, tgt_mask_torch, tgt_edge_torch, rgb_frames, aolp_frames, dolp_frames, meta_dict in tqdm(train_loader):
            # gradient initialization
            optimizer.zero_grad()
            # preprocessing
            input_rgb = preprocessing.feature_pyramid_extract(rgb_frames.cuda())
            input_aolp = preprocessing.feature_pyramid_extract(aolp_frames.cuda())
            input_dolp = preprocessing.feature_pyramid_extract(dolp_frames.cuda())
            input_rgb = (tgt_image_torch.cuda(), input_rgb['query_featmap'], input_rgb['supp_featmap'], input_rgb['opflow_angle_mag'])
            input_aolp = (tgt_aolps_torch.cuda(), input_aolp['query_featmap'], input_aolp['supp_featmap'], input_aolp['opflow_angle_mag'])
            input_dolp = (tgt_dolps_torch.cuda(), input_dolp['query_featmap'], input_dolp['supp_featmap'], input_dolp['opflow_angle_mag'])
            
            output_dict = model(input_rgb, input_aolp, input_dolp)
            
            losses = compute_loss(output_dict, tgt_mask_torch, final_weight=2.0, verbose=False, epoch=epoch)
            
            losses.backward()
            optimizer.step()
            
            # recode
            train_loss_iter.append(losses.detach().item())
            train_score_iter.append(metrics_fn(torch.sigmoid(output_dict['AE1']).cpu(), tgt_mask_torch.int()).item())
            
        avg_iter_loss = np.mean(train_loss_iter)
        avg_iter_score = np.mean(train_score_iter)
        print(f"train loss: {avg_iter_loss:.5f}, score: {avg_iter_score:.5f}")
        # recode 
        train_loss_epochs.append(avg_iter_loss)
        train_score_epochs.append(avg_iter_score)
        ''' validation '''
        model.eval()
        val_loss_iter = []
        val_score_iter = []

        for tgt_image_torch, tgt_aolps_torch, tgt_dolps_torch, tgt_mask_torch, tgt_edge_torch, rgb_frames, aolp_frames, dolp_frames, meta_dict in tqdm(val_loader):
            # Preprocessing
            input_rgb = preprocessing.feature_pyramid_extract(rgb_frames.cuda())
            input_aolp = preprocessing.feature_pyramid_extract(aolp_frames.cuda())
            input_dolp = preprocessing.feature_pyramid_extract(dolp_frames.cuda())
            input_rgb = (tgt_image_torch.cuda(), input_rgb['query_featmap'], input_rgb['supp_featmap'], input_rgb['opflow_angle_mag'])
            input_aolp = (tgt_aolps_torch.cuda(), input_aolp['query_featmap'], input_aolp['supp_featmap'], input_aolp['opflow_angle_mag'])
            input_dolp = (tgt_dolps_torch.cuda(), input_dolp['query_featmap'], input_dolp['supp_featmap'], input_dolp['opflow_angle_mag'])

            # Forward pass
            with torch.no_grad():
                output_dict = model(input_rgb, input_aolp, input_dolp)
            
            # 検出結果の確認用を保存
            filename = os.path.basename(meta_dict['tgt_image_path'][0])
            plt.subplot(2,2,1)
            tgt_img = Image.open(meta_dict['tgt_image_path'][0])
            plt.imshow(tgt_img)
            plt.title("target image")
            plt.subplot(2,2,2)
            supp_img = Image.open(meta_dict['supp_image_path'][0])
            plt.imshow(supp_img)
            plt.title("support image")
            plt.subplot(2,2,3)
            plt.imshow(tgt_mask_torch.squeeze().cpu().numpy(), cmap='gray')
            plt.title("target mask")
            plt.subplot(2,2,4)
            plt.imshow(torch.sigmoid(output_dict['opflow_rgb']).squeeze().cpu().numpy(), cmap='gray')
            plt.title("output mask")
            plt.savefig(os.path.join(dir_val_outputs, f"epoch{epoch}_{filename}"))
            
            
            # Loss computation
            losses = compute_loss(output_dict, tgt_mask_torch, verbose=True, epoch=epoch)
            score = metrics_fn(torch.sigmoid(output_dict['AE1']).cpu(), tgt_mask_torch.int())

            # Record metrics
            val_loss_iter.append(losses.item())
            val_score_iter.append(score.item())

        # Average metrics for the validation epoch
        avg_iter_loss = np.mean(val_loss_iter)
        avg_iter_score = np.mean(val_score_iter)
        print(f"Validation loss: {avg_iter_loss:.5f}, score: {avg_iter_score:.5f}")
        
        # Record epoch-level metrics
        val_loss_epochs.append(avg_iter_loss)
        val_score_epochs.append(avg_iter_score)

        # Save validation results periodically
        # if epoch % 5 == 0:
        save_plots(train_loss_epochs, 
                train_score_epochs,
                val_loss_epochs,  
                val_score_epochs,  
                save_path=os.path.join(dir_loss_metrics_graph, f"learning_curve+{args.phase}.png"))

        # Early stopping check
        # es(-avg_iter_score, model)
        es(avg_iter_loss, model)
        if es.early_stop:
            print("Early Stopping.")
            break


if __name__ == "__main__":
    main()