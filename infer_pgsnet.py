import os
import pdb
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryFBetaScore
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from sklearn.metrics import mean_absolute_error

from config import parse_args, create_mirror_dataset
from metrics import get_maxFscore_and_threshold

from preprocessing import PreProcessing


def test_model(model, test_loader, args, result_dir):
    
    # 損失関数と評価指標の設定
    metrics_fn = BinaryFBetaScore(beta=0.5)
    
    # focal_loss_fn = FocalLoss(mode="binary")
    dice_loss_fn = DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def compute_loss(output, tgt_mask_torch):
        return sum([
            bce_loss_fn(output[i], tgt_mask_torch.cuda()) + dice_loss_fn(output[i], tgt_mask_torch.cuda())
            for i in range(len(output))
        ])
    
    # preprocessing = PreProcessing(grid_size=args.grid_size)
    
    val_loss_iter = []
    val_score_iter = []
    val_mae_iter = []
    val_maxFbeta_iter = []
    
    dir_val_outputs = result_dir
    dir_thresholded_outputs = os.path.join(dir_val_outputs, "thresholded_outputs")
    dir_visualization_outputs = os.path.join(dir_val_outputs, "visualization_outputs")
    results_txt_path = os.path.join(dir_val_outputs, "results.txt")
    os.makedirs(dir_val_outputs, exist_ok=True)
    os.makedirs(dir_thresholded_outputs, exist_ok=True)
    os.makedirs(dir_visualization_outputs, exist_ok=True)
    
    for tgt_image_torch, tgt_aolps_torch, tgt_dolps_torch, tgt_mask_torch, tgt_edge_torch, rgb_frames, aolp_frames, dolp_frames, meta_dict in tqdm(test_loader):
        
        with torch.no_grad():
            
            pred = model(tgt_image_torch.cuda(), tgt_aolps_torch.cuda(), tgt_dolps_torch.cuda())
            
            loss = compute_loss(pred, tgt_mask_torch)
        
            final_pred = torch.sigmoid(pred[-1]) # final prediction
            score = metrics_fn(final_pred.float().cpu(), tgt_mask_torch.int())
            maxFbeta, thres = get_maxFscore_and_threshold(tgt_mask_torch.squeeze().numpy().flatten(), final_pred.flatten().cpu())
            print(f"Max F-beta Score: {maxFbeta:.5f}, Threshold: {thres:.5f}")
            mae = mean_absolute_error(tgt_mask_torch.cpu().numpy().flatten(), final_pred.cpu().numpy().flatten())
            
            tgt_img_path = meta_dict["tgt_image_path"][0]
            tgt_aolp_path = meta_dict["tgt_aolp_path"][0]
            tgt_dolp_path = meta_dict["tgt_dolp_path"][0]
            mask_path = meta_dict["tgt_mask_path"][0]
            supp_img_path = meta_dict["supp_image_path"][0]
            supp_aolp_path = meta_dict["supp_aolp_path"][0]
            supp_dolp_path = meta_dict["supp_dolp_path"][0]
            
            print("Image path: ", tgt_img_path, "Loss: ", round(loss.item(), 3), "Score: ", round(score.item(), 3), "MAE: ", round(mae, 3))
            val_loss_iter.append(loss.item())
            val_score_iter.append(score.item())
            val_mae_iter.append(mae)
            val_maxFbeta_iter.append(maxFbeta)
            
            # Save predicted mask
            tgt_rgb_img = Image.open(tgt_img_path).convert('RGB')
            tgt_aolp_img = Image.open(tgt_aolp_path).convert('RGB')
            tgt_dolp_img = Image.open(tgt_dolp_path).convert('RGB')
            supp_rgt_img = Image.open(supp_img_path).convert('RGB')
            supp_aolp_img = Image.open(supp_aolp_path).convert('RGB')
            supp_dolp_img = Image.open(supp_dolp_path).convert('RGB')
            mask_img = Image.open(mask_path).convert('L')
            w, h = tgt_rgb_img.size
            pred_masks = [torch.sigmoid(p.cpu()) for p in pred]
            pred_masks_resized = [transforms.Resize((h, w))(p) for p in pred_masks]
            
            # Apply threshold to visualize binary prediction
            thresholded_pred = (final_pred.cpu() > thres).float()
            thresholded_pred_resized = transforms.Resize((h, w))(thresholded_pred)
            
            # Save thresholded prediction as an image (0-255) in PNG format
            thresholded_pred_img = (thresholded_pred_resized.numpy() * 255).astype(np.uint8)
            
            base_name = os.path.basename(os.path.abspath(os.path.join(tgt_img_path, "../../")))
            base_name = base_name + "_" + os.path.splitext(os.path.basename(tgt_img_path))[0]

            thresholded_output_path = os.path.join(dir_thresholded_outputs, f"{base_name}_thresholded_{maxFbeta:.3f}.png")
            Image.fromarray(thresholded_pred_img.squeeze()).save(thresholded_output_path)
                    
            # サブプロットを作成
            num_imgs = 8 + len(pred_masks_resized) + 1  # 8は最初の画像数、+1は最後の画像
            col = 4
            row = (num_imgs + col - 1) // col  # 行数を計算

            fig, axes = plt.subplots(row, col, figsize=(8, 12))
            fig.set_dpi(300)
            axes = axes.flatten()  # 1次元配列に変換

            # 最初の画像をプロット
            axes[0].imshow(tgt_rgb_img)
            axes[0].set_title("RGB Image")
            axes[0].axis('off')

            axes[1].imshow(tgt_aolp_img)
            axes[1].set_title("AoLP Image")
            axes[1].axis('off')

            axes[2].imshow(tgt_dolp_img)
            axes[2].set_title("DoLP Image")
            axes[2].axis('off')

            axes[3].imshow(mask_img, cmap='gray')
            axes[3].set_title("Ground Truth")
            axes[3].axis('off')

            axes[4].imshow(supp_rgt_img)
            axes[4].set_title("Supplementary RGB Image")
            axes[4].axis('off')

            axes[5].imshow(supp_aolp_img)
            axes[5].set_title("Supplementary AoLP Image")
            axes[5].axis('off')

            axes[6].imshow(supp_dolp_img)
            axes[6].set_title("Supplementary DoLP Image")
            axes[6].axis('off')

            # 予測マスクのプロット
            start_idx = 7  # 予測マスクの開始インデックス
            for i, pred_mask_resized in enumerate(pred_masks_resized):
                idx = start_idx + i
                axes[idx].imshow(pred_mask_resized.squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[idx].set_title(f"Predicted Mask {i+1}")
                axes[idx].axis('off')

            # Thresholded Prediction のプロット
            axes[start_idx + len(pred_masks_resized)].imshow(thresholded_pred_resized.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[start_idx + len(pred_masks_resized)].set_title("Thresholded Prediction")
            axes[start_idx + len(pred_masks_resized)].axis('off')

            # 不要なサブプロットを非表示にする
            for ax in axes[start_idx + len(pred_masks_resized) + 1:]:
                ax.axis('off')

            # 図全体のタイトルを設定
            plt.suptitle(f"Max F-beta: {maxFbeta:.3f}, Thres: {thres:.3f}", fontsize=16)

            # レイアウトを調整
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # タイトルとサブプロットが重ならないように

            # 画像を保存
            visualization_output_file = os.path.join(dir_visualization_outputs, f"{base_name}_visualization_{maxFbeta:.3f}.png")
            plt.savefig(visualization_output_file)
            plt.close()
    
    # all test data average
    avg_loss = np.mean(val_loss_iter)
    avg_score = np.mean(val_score_iter)
    avg_mae = np.mean(val_mae_iter)
    avg_maxFbeta = np.mean(val_maxFbeta_iter)
    print(f"\nAverage Test Loss: {avg_loss:.5f}, Average Test Score: {avg_score:.5f}, Average MAE: {avg_mae:.5f}, Average Max F-beta: {avg_maxFbeta:.5f}")

    with open(results_txt_path, "w") as f:
        f.write(f"Average Test Loss: {avg_loss:.5f}\n")
        f.write(f"Average Test Score: {avg_score:.5f}\n")
        f.write(f"Average MAE: {avg_mae:.5f}\n")
        f.write(f"Average Max F-beta: {avg_maxFbeta:.5f}\n")


def main():
    args = parse_args()
    if args.batch_size != 1:
        print("Batch size must be 1.")
        args.batch_size = 1
    train_loader, val_loader, test_loader = create_mirror_dataset(args)
    data_type = {"train": train_loader, "val": val_loader, "test": test_loader}
    
    # Load trained model
    pram_path = os.path.join(args.result_dir, "ckpt", "best_weight.pth")
    print("Model parameter path: ", pram_path)
    assert os.path.exists(pram_path), f"Model path {pram_path} does not exist."
    
    # model instance
    netfile = importlib.import_module("model." + args.model)
    model = netfile.Network(in_dim=3).cuda()
    model.load_state_dict(torch.load(pram_path, weights_only=True))
    model.eval()
    
    # Test the model
    for d_type, loader in data_type.items():
        print("which train validation or test -> ", d_type)
        # if d_type != "test":
        #     print("Skip train and validation.")
        #     continue
        result_dir = os.path.join(args.result_dir, d_type + "_outputs")
        test_model(model, loader, args, result_dir)


if __name__ == "__main__":
    main()
