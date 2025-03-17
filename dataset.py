import os
import glob
import pdb
import random

from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, root_dir, train_mode: bool):
        """
        Args:
            root_dir (str): データセットのルートディレクトリ
            transform (callable, optional): データ変換を指定する関数または操作。デフォルトはNone
        """
        self.root_dir = root_dir
        self.train_mode = train_mode
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
        ])
        
        self.videos = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        meta_dict = {}
        
        ano_dir = os.path.join(self.videos[idx], "anotation")
        json_files = glob.glob(os.path.join(ano_dir, "*.json"))
        txt_files = glob.glob(os.path.join(ano_dir, "*.txt"))

        if not json_files or not txt_files:
            raise FileNotFoundError(f"Missing JSON or TXT file in {ano_dir}")

        json_path = json_files[0]
        txt_path = txt_files[0]
        
        # ターゲット画像とソース画像のフレームIDを取得
        tgt_frm_id = os.path.splitext(os.path.basename(json_path))[0].replace("_rgb", "")
        with open(txt_path, "r") as f:
            supp_frm_id = f.read().strip() 
        # target image path
        tgt_image_path = os.path.join(self.videos[idx], "RGB", tgt_frm_id + "_rgb.jpg")
        tgt_aolp_path = os.path.join(self.videos[idx], "AoLP", tgt_frm_id + "_aolp.tiff")
        tgt_dolp_path = os.path.join(self.videos[idx], "DoLP", tgt_frm_id + "_dolp.tiff")
        tgt_aolp_vis_path = os.path.join(self.videos[idx], "Visual", tgt_frm_id + "_aolp_crop.jpg")
        tgt_dolp_vis_path = os.path.join(self.videos[idx], "Visual", tgt_frm_id + "_dolp_crop.jpg")
        tgt_mask_path = os.path.join(self.videos[idx], "mask", tgt_frm_id + "_rgb.png")
        # source image path
        supp_image_path = os.path.join(self.videos[idx], "RGB", supp_frm_id + "_rgb.jpg")
        supp_aolp_vis_path = os.path.join(self.videos[idx], "Visual", supp_frm_id + "_aolp_crop.jpg")
        supp_dolp_vis_path = os.path.join(self.videos[idx], "Visual", supp_frm_id + "_dolp_crop.jpg")
        
        tgt_image = Image.open(tgt_image_path).convert("RGB")  # RGB形式に変換
        supp_image = Image.open(supp_image_path).convert("RGB")
        mask = Image.open(tgt_mask_path).convert("L")  # グレースケールに変換
        tgt_aolp = Image.open(tgt_aolp_path)  # raw data 0 ~ pi
        tgt_dolp = Image.open(tgt_dolp_path)  # raw data 0 ~ 1
        tgt_aolp_vis = Image.open(tgt_aolp_vis_path).convert("RGB")
        tgt_dolp_vis = Image.open(tgt_dolp_vis_path).convert("RGB")
        supp_aolp_vis = Image.open(supp_aolp_vis_path).convert("RGB")
        supp_dolp_vis = Image.open(supp_dolp_vis_path).convert("RGB")
        
        # image transformation
        input_rgb_transformed = self.img_transform(tgt_image)
        input_mask_transformed = self.mask_transform(mask)
        input_mask_transformed = (input_mask_transformed > 0).to(torch.uint8)
        
        input_dolp_transformed = torch.stack(
            [self.mask_transform(i).squeeze(0) for ind, i in enumerate(crop(tgt_dolp)) if ind != 2],
            axis=0
        )
        input_aolp_transformed = torch.stack(
            [self.mask_transform(np.sin(np.deg2rad(i))).squeeze(0) for ind, i in enumerate(crop(tgt_aolp)) if ind != 2],
            axis=0
        )
        
        # 2 frames
        input_rgb_frames = torch.stack([self.mask_transform(tgt_image), self.mask_transform(supp_image)], dim=0) * 255.0
        input_aolp_frames = torch.stack([self.mask_transform(tgt_aolp_vis), self.mask_transform(supp_aolp_vis)], dim=0) * 255.0
        input_dolp_frames = torch.stack([self.mask_transform(tgt_dolp_vis), self.mask_transform(supp_dolp_vis)], dim=0) * 255.0
        
        # target ground truth edge
        tgt_mask = cv2.imread(tgt_mask_path, cv2.IMREAD_GRAYSCALE)  # グレースケールとして読み込む
        tgt_mask = (tgt_mask > 0).astype(np.float32)  # 二値化
        low_threshold, high_threshold = 50, 150  # 任意の値を設定
        edge = cv2.Canny((tgt_mask * 255).astype(np.uint8), low_threshold, high_threshold)
        tgt_edge_torch = torch.tensor(np.expand_dims(edge, 0), dtype=torch.float32)
        tgt_edge_torch = (tgt_edge_torch > 0).to(torch.uint8)
        
        if self.train_mode and random.random() > 0.5:
            input_rgb_transformed = transforms.functional.hflip(input_rgb_transformed)
            input_mask_transformed = transforms.functional.hflip(input_mask_transformed)
            input_aolp_transformed = transforms.functional.hflip(input_aolp_transformed)
            input_dolp_transformed = transforms.functional.hflip(input_dolp_transformed)
            input_rgb_frames = transforms.functional.hflip(input_rgb_frames)
            input_aolp_frames = transforms.functional.hflip(input_aolp_frames)
            input_dolp_frames = transforms.functional.hflip(input_dolp_frames)
            tgt_edge_torch = transforms.functional.hflip(tgt_edge_torch)

        # メタデータ
        meta_dict["tgt_image_path"] = tgt_image_path
        meta_dict["tgt_aolp_path"] = tgt_aolp_vis_path
        meta_dict["tgt_dolp_path"] = tgt_dolp_vis_path
        meta_dict["tgt_mask_path"] = tgt_mask_path
        meta_dict["supp_image_path"] = supp_image_path
        meta_dict["supp_aolp_path"] = supp_aolp_vis_path
        meta_dict["supp_dolp_path"] = supp_dolp_vis_path
        meta_dict["hflip"] = self.train_mode and random.random() > 0.5
        
        return input_rgb_transformed, input_aolp_transformed, input_dolp_transformed, input_mask_transformed, tgt_edge_torch, input_rgb_frames, input_aolp_frames, input_dolp_frames, meta_dict


def crop(img_pil: Image) -> list:
    """Crop image into 4 segments."""
    img = np.array(img_pil)
    segments = []
    h, w = img.shape[:2]
    half_h, half_w = h //2, w // 2
    for i in range(0, h, half_h):
        for j in range(0, w, half_w):
            segments.append(img[i:i+half_h, j:j+half_w])
    return segments # 1-> red, 2-> green, 3-> green, 4-> blue


def pil_to_tensor_keep_range(pil_image):
    # NumPy 配列に変換してから torch.Tensor に変換
    numpy_array = np.array(pil_image)
    tensor_image = torch.from_numpy(numpy_array).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return tensor_image

