from argparse import ArgumentParser
import pdb
import os
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import numpy as np

from dataset import VideoDataset
from dataset_VMD import VMD_Dataset
from dataset_PMD import PMD_Dataset

def parse_args():
    '''コマンドライン引数'''
    parser = ArgumentParser()
    
    # network setting
    parser.add_argument('-model', type=str, default='proposed_net', help='model name')
    
    parser.add_argument('-dataset_path', type=str, required=True, help="dataset path") 
    parser.add_argument('-result_dir', type=str, required=True, help="dataset path") 
    parser.add_argument('-phase', type=int) 
    parser.add_argument('--from_scratch',action='store_true')
    parser.add_argument('-g', '--grid_size', type=int, default=52)
    parser.add_argument('--new_model', action='store_true')
    parser.add_argument('--vmd', action='store_true')
    # learning setting
    parser.add_argument('-random_seed', type=int, default=42, help='Defalut:42') 
    parser.add_argument('-epochs', type=int, default=150, help='defalut:150')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('-batch_size', type=int, default=4, help='Defalut:4')
    parser.add_argument('-resize', type=int, default=416 , help='Default:416')
    parser.add_argument("-patient", type=int, default=10, help="Early Stopping . the number of epoch. defalut 10")
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 regularization) strength')
    # test
    parser.add_argument('-model_name', type=str, default="PGSNet", help="model name")
    parser.add_argument('-mirror_type', type=str, default=None, help="mirror type")
    parser.add_argument('--test_only', action='store_true')
    
    return parser.parse_args()

    

def create_mirror_dataset(args):
    """
    指定されたデータセットパスからミラー検出用のトレーニング、検証、テスト用データセットを作成します。
    
    Args:
        args: 引数オブジェクト。以下の属性が必要:
            - dataset_path (str): データセットのルートディレクトリ
            - mirror_type (str): テストに使用するミラータイプ
            - resize (int): 画像のリサイズサイズ
            - random_seed (int): データ分割のためのシード値
            - batch_size (int): バッチサイズ
            
    Returns:
        train_loader (DataLoader): トレーニング用データローダー
        val_loader (DataLoader): 検証用データローダー
        test_loader (DataLoader): テスト用データローダー
    """
    # データセットパスとミラータイプの確認
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    if args.mirror_type is None:
        raise ValueError("No test directory specified for mirror_type.")
    
    # データセット内のミラータイプのディレクトリを取得
    mirror_types = os.listdir(args.dataset_path)
    
    train_datasets = []
    val_datasets = []
    test_set = None
    
    # データセットの分割
    for mirror_type in mirror_types:
        dataset_path = os.path.join(args.dataset_path, mirror_type)
        
        # テストデータセットの準備
        if args.mirror_type in mirror_type:
            test_set = VideoDataset(dataset_path, train_mode=False)
            print(f"Test dataset ({mirror_type}): {len(test_set)} samples")
        else:
            # トレーニングおよび検証データセットの準備
            train_set = VideoDataset(dataset_path, train_mode=False)
            # トレーニングと検証データの分割
            train_size, val_size = getTrainTestCounts(train_set)
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_set, [train_size, val_size],
                generator=torch.Generator().manual_seed(args.random_seed)
            )
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            
            print(f"Train dataset ({mirror_type}): {len(train_dataset)} train, {len(val_dataset)} valid")
    
    # トレーニングと検証データセットの結合
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    print(f"Total - Train: {len(train_dataset)}, Valid: {len(val_dataset)}, Test: {len(test_set) if test_set else 0}")

    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, pin_memory=True) if test_set else None

    return train_loader, val_loader, test_loader


def create_RGBmirror_dataset(dataset_name: str, args):
    if dataset_name == "VMD":
        dataset_path = "/data2/yoshimura/mirror_detection/DATA/VMD"
        train_dataset = VMD_Dataset(os.path.join(dataset_path, "train"), scale=args.resize)
        test_dataset = VMD_Dataset(os.path.join(dataset_path, "test"), scale=args.resize)
    elif dataset_name == "PMD":
        dataset_path = "/data2/yoshimura/mirror_detection/DATA/PMD"
        train_dataset = PMD_Dataset(os.path.join(dataset_path, "train"), scale=args.resize)
        test_dataset = PMD_Dataset(os.path.join(dataset_path, "test", "_ALL"), scale=args.resize)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    dataset = ConcatDataset([train_dataset, test_dataset])
    train_size, val_size = getTrainTestCounts(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f" train: {train_dataset.__len__()},\n val: {val_dataset.__len__()},\n")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_lodaer = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    return train_loader, val_lodaer, None


def get_dataloader(dataset: Dataset, batch_size: int = 32) -> DataLoader:
    train_size, val_size = getTrainTestCounts(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f" train: {train_dataset.__len__()},\n val: {val_dataset.__len__()},\n")
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0, pin_memory= True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,num_workers=0, pin_memory= True)
    return train_loader, val_loader

def getTrainTestCounts(dataset: Dataset) -> tuple:
    train_size = int(dataset.__len__() * 0.8) 
    val_size   = dataset.__len__() - train_size
    return train_size, val_size


def set_seed(seed=42):
    random.seed(seed)                    # Python の乱数シード
    np.random.seed(seed)                 # NumPy の乱数シード
    torch.manual_seed(seed)              # PyTorch の乱数シード（CPU）
    torch.cuda.manual_seed(seed)         # PyTorch の乱数シード（GPU）
    torch.cuda.manual_seed_all(seed)     # PyTorch の乱数シード（複数GPU）

    # 再現性のために、以下の設定を有効にします。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


