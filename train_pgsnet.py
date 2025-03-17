import os
import pdb
import importlib

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import BinaryFBetaScore
import matplotlib.pyplot as plt
import numpy as np

from earlystop import EarlyStopping
from config import set_seed, parse_args, create_mirror_dataset
from metrics import get_maxFscore_and_threshold
from plot import save_plots, plot_and_save


def main():
    
    filepath = os.path.abspath(__file__)
    print("Executing file:", filepath)
    
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

    # loss function
    dice_loss_fn = DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def compute_loss(output, tgt_mask_torch):
        return sum([
            bce_loss_fn(output[i], tgt_mask_torch.float().cuda()) + dice_loss_fn(output[i], tgt_mask_torch.cuda())
            for i in range(len(output))
        ])
    
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
            output = model(tgt_image_torch.cuda(), tgt_aolps_torch.cuda(), tgt_dolps_torch.cuda())
            
            losses = compute_loss(output, tgt_mask_torch)
            losses.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()
            
            # recode
            train_loss_iter.append(losses.detach().item())
            train_score_iter.append(metrics_fn(torch.sigmoid(output[-1]).cpu(), tgt_mask_torch.int()).item())
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
            
            # Forward pass
            with torch.no_grad():
                output = model(tgt_image_torch.cuda(), tgt_aolps_torch.cuda(), tgt_dolps_torch.cuda())

            # Loss computation
            losses = compute_loss(output, tgt_mask_torch)

            score = metrics_fn(torch.sigmoid(output[-1]).cpu(), tgt_mask_torch.int())
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