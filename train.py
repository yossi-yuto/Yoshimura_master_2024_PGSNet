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
from config import set_seed, parse_args, create_mirror_dataset, yvos_dataloader, create_RGBmirror_dataset
from metrics import get_maxFscore_and_threshold
from plot import save_plots, plot_and_save
from layer_freeze import freeze_out_layer
from model.proposed_net import Network


def load_and_freeze_pretrained_parameters(model, pretrained_path):
    """
    事前学習済みのパラメータを読み込み、一致するパラメータをモデルにロードし、
    そのパラメータのみフリーズする関数。

    Args:
        model (torch.nn.Module): モデルオブジェクト。
        pretrained_path (str): 事前学習済みパラメータのファイルパス。
    """
    # 事前学習済みのパラメータを読み込む
    pretrained_dict = torch.load(pretrained_path, weights_only=False)
    
    # 現在のモデルのパラメータ辞書を取得
    model_dict = model.state_dict()
    
    # 一致するパラメータのみコピー
    matched_params = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # パラメータのロードとフリーズ
    print("Loading and freezing the following parameters:")
    for name, param in model.named_parameters():
        if name in matched_params:
            # パラメータをロード
            param.data.copy_(matched_params[name])
            # フリーズ
            param.requires_grad = False
        
        if param.requires_grad:
            print(f"  - {name} (trainable)")
        else:
            print(f"  - {name} (frozen)")
    
    # 未ロードのパラメータは学習可能なまま
    print("\nAll other layers are set to trainable.")
    
    
def load_pretrained_parameters_and_freeze(model, pretrained_path):
    """
    Conformerモデルに事前学習済みパラメータを読み込み、
    特定の層（conv1 と bn1）以外をフリーズする関数。

    Args:
        model (torch.nn.Module): Conformerモデルオブジェクト。
        pretrained_path (str): 事前学習済みパラメータのパス。
    """
    # 1. 事前学習済みパラメータをロード
    pretrained_dict = torch.load(pretrained_path)
    
    # 2. 現在のモデルのパラメータ辞書を取得
    model_dict = model.state_dict()
    
    # 3. 一致するパラメータのみを事前学習済みパラメータからコピー
    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # 4. モデルにパラメータをロード
    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict)
    
    # 5. 全てのパラメータをフリーズ
    for param in model.parameters():
        param.requires_grad = False

    # 6. `conv1` と `bn1` のみ再度学習可能に設定
    for name, param in model.named_parameters():
        if 'conv1' in name or 'bn1' in name:
            param.requires_grad = True
            print(f"Layer {name} is set to trainable.")
        else:
            print(f"Layer {name} is frozen.")

    print("Model parameters loaded and layers frozen as specified.")


def update_model_with_conformer_params(model, dir_checkpoint):
    """
    Update the model's state_dict with parameters from conformer files and disable gradients for those parameters.
    Args:
        model (torch.nn.Module): The model to update.
        dir_checkpoint (str): The directory where the conformer checkpoint files are located.
    """
    model_dict = model.state_dict()
    conformer_files = ["rgb_backbone.pth", "aolp_backbone.pth", "dolp_backbone.pth"]
    
    for file in conformer_files:
        checkpoint_path = os.path.join(dir_checkpoint, file)
        try:
            conformer_param = torch.load(checkpoint_path)
        except FileNotFoundError:
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        
        filtered_conformer_param = {k: v for k, v in conformer_param.items() if 'conformer' in k or 'early_fusion' in k}
        model_dict.update(filtered_conformer_param)
    
    model.load_state_dict(model_dict)
    
    for name, param in model.named_parameters():
        if 'conformer' in name or 'early_fusion' in name:
            param.requires_grad = False


def model_conformer_reading_freeze(model: torch.nn.Module, pretrained_param_dict):
    model_dict = model.state_dict()
    
    filtered_pretrained_param_dict = {k: v for k, v in pretrained_param_dict.items() if 'conformer' in k or 'early_fusion' in k}
    model_dict.update(filtered_pretrained_param_dict)
    
    for name, param in model.named_parameters():
        if name in filtered_pretrained_param_dict:
            param.requires_grad = False
            
    model.load_state_dict(model_dict)
    return model


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
    
    # weight init
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
    
    # encoder freeze
    pretrained_param_path = "/data2/yoshimura/mirror_detection/PGSNet/pretrain_param/Conformer_base_patch16.pth"
    model.conformer1.load_state_dict(torch.load(pretrained_param_path, weights_only=True), strict=False)
    model.conformer2.load_state_dict(torch.load(pretrained_param_path, weights_only=True), strict=False)
    model.conformer3.load_state_dict(torch.load(pretrained_param_path, weights_only=True), strict=False)
    for name, param in model.named_parameters():
        if "conformer" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # preprocessing setting
    print("Preprocessing setting...")
    preprocessing = PreProcessing(grid_size=args.grid_size)

    # loss function
    dice_loss_fn = DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def compute_loss(output_dict: dict, tgt_mask_torch: torch.Tensor, final_weight: float = 1.0, verbose: bool = False) -> list:
        losses = []
        tgt_mask_cuda = tgt_mask_torch.cuda()  # 1回だけ CUDA に送る
        for key, value in output_dict.items():
            loss = bce_loss_fn(value, tgt_mask_cuda.to(torch.float32)) + dice_loss_fn(value, tgt_mask_cuda)
            if key == "AE1":
                losses.append(loss * final_weight)
            else:
                losses.append(loss)
            
            if verbose:
                print(f"{key} loss: {loss.item()}")  # 個別の損失を表示
        return losses
    
    # metrics
    metrics_fn = BinaryFBetaScore(beta=0.5)
    
    # early stopping
    es = EarlyStopping(verbose=True, 
                        patience=args.patient, 
                        filepath=os.path.join(dir_checkpoint, "best_weight.pth"))
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    
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
            
            losses = compute_loss(output_dict, tgt_mask_torch, final_weight=2.0, verbose=False)
            losses = sum(losses)
            
            losses.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()
            
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

            # Loss computation
            losses = compute_loss(output_dict, tgt_mask_torch, verbose=True)
            losses = sum(losses)

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
        if epoch % 5 == 0:
            save_plots(train_loss_epochs, 
                    train_score_epochs,
                    val_loss_epochs,  
                    val_score_epochs,  
                    save_path=os.path.join(dir_loss_metrics_graph, "learning_curve.png"))

        # Early stopping check
        es(avg_iter_loss, model)
        if es.early_stop:
            print("Early Stopping.")
            break


# 学習可能なパラメータを表示する関数
def print_params(model):
    print('学習可能なパラメータ')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, "freezed layer")
    print("===================================")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "training layer")


if __name__ == "__main__":
    main()