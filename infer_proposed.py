import os
import pdb
import importlib

import torch
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
from metrics import calc_iou_multi_thresh, get_maxFscore_and_threshold_multi_beta

from preprocessing import PreProcessing

def resize_predictions(predictions :dict, target_size: tuple) -> dict:
    resized_preds = {}
    for key, value in predictions.items():
        resized_preds[key] = transforms.Resize(target_size)(torch.sigmoid(value).cpu())
    return resized_preds


def test_model(model, test_loader, args, result_dir):
    
    focal_loss_fn = FocalLoss(mode="binary")
    dice_loss_fn = DiceLoss(mode="binary")
    # bce_loss_fn = nn.BCEWithLogitsLoss()
    def compute_loss(output_dict: dict, tgt_mask_torch: torch.Tensor, final_weight: float = 1.0, verbose: bool = False) -> list:
        losses = []
        tgt_mask_cuda = tgt_mask_torch.cuda()  # 1回だけ CUDA に送る
        for key, value in output_dict.items():
            loss = focal_loss_fn(value, tgt_mask_cuda) + dice_loss_fn(value, tgt_mask_cuda)
            if key == "AE1":
                losses.append(loss * final_weight)
            else:
                losses.append(loss)
            
            if verbose:
                print(f"{key} loss: {loss.item()}")  # 個別の損失を表示
        return losses
    
    preprocessing = PreProcessing(grid_size=args.grid_size)
    
    loss_list :list = []
    MAE_list :list = []
    IoU_list :dict = {
        "IoU_threshed_01": [],
        "IoU_threshed_03": [],
        "IoU_threshed_05": [],
        "IoU_threshed_07": [],
    }
    maxFbeta_list :dict = {
        "maxFbeta_beta_03": [],
        "maxFbeta_beta_05": [],
        "maxFbeta_beta_1": [],
    }
    
    dir_val_outputs = result_dir
    dir_thresholded_outputs = os.path.join(dir_val_outputs, "thresholded_outputs")
    dir_visualization_outputs = os.path.join(dir_val_outputs, "visualization_outputs")
    results_txt_path = os.path.join(dir_val_outputs, "results.txt")
    os.makedirs(dir_val_outputs, exist_ok=True)
    os.makedirs(dir_thresholded_outputs, exist_ok=True)
    os.makedirs(dir_visualization_outputs, exist_ok=True)
    
    for tgt_image_torch, tgt_aolps_torch, tgt_dolps_torch, tgt_mask_torch, tgt_edge_torch, rgb_frames, aolp_frames, dolp_frames, meta_dict in tqdm(test_loader):
        
        with torch.no_grad():
            input_rgb = preprocessing.feature_pyramid_extract(rgb_frames.cuda())
            input_aolp = preprocessing.feature_pyramid_extract(aolp_frames.cuda())
            input_dolp = preprocessing.feature_pyramid_extract(dolp_frames.cuda())
            input_rgb = (tgt_image_torch.cuda(), input_rgb['query_featmap'], input_rgb['supp_featmap'], input_rgb['opflow_angle_mag'])
            input_aolp = (tgt_aolps_torch.cuda(), input_aolp['query_featmap'], input_aolp['supp_featmap'], input_aolp['opflow_angle_mag'])
            input_dolp = (tgt_dolps_torch.cuda(), input_dolp['query_featmap'], input_dolp['supp_featmap'], input_dolp['opflow_angle_mag'])
            
            pred :dict = model(input_rgb, input_aolp, input_dolp)
            
            loss = compute_loss(pred, tgt_mask_torch, verbose=False)
            loss = sum(loss)
            loss_list.append(loss.cpu().item())
        
            final_pred = torch.sigmoid(pred['AE1']) # final prediction
            print(f"final_pred max value: {final_pred.max().cpu().item():.5f}")
            
            mae = mean_absolute_error(tgt_mask_torch.cpu().numpy().flatten(), final_pred.cpu().numpy().flatten())
            MAE_list.append(mae)
            maxFbetas :dict = get_maxFscore_and_threshold_multi_beta(tgt_mask_torch.squeeze().numpy().flatten(), final_pred.flatten().cpu())
            maxFbeta_list["maxFbeta_beta_03"].append(maxFbetas[0.3]['max_fscore'])
            maxFbeta_list["maxFbeta_beta_05"].append(maxFbetas[0.5]['max_fscore'])
            maxFbeta_list["maxFbeta_beta_1"].append(maxFbetas[1.0]['max_fscore'])
            IoU_multi_threshed :dict = calc_iou_multi_thresh(tgt_mask_torch.cpu().numpy().flatten(), final_pred.cpu().numpy().flatten(), thresholds=[0.1, 0.3, 0.5, 0.7])
            # pdb.set_trace()
            IoU_list["IoU_threshed_01"].append(IoU_multi_threshed[0.1])
            IoU_list["IoU_threshed_03"].append(IoU_multi_threshed[0.3])
            IoU_list["IoU_threshed_05"].append(IoU_multi_threshed[0.5])
            IoU_list["IoU_threshed_07"].append(IoU_multi_threshed[0.7])
            
            print(f"Loss: {loss:.5f}, MAE: {mae:.5f}")
            print(f"Max F-beta Scores:")
            print(f"  - β=0.3: {maxFbetas[0.3]['max_fscore']:.5f}, precision: {maxFbetas[0.3]['precision']:.5f}, recall: {maxFbetas[0.3]['recall']:.5f}")
            print(f"  - β=0.5: {maxFbetas[0.5]['max_fscore']:.5f}, precision: {maxFbetas[0.5]['precision']:.5f}, recall: {maxFbetas[0.5]['recall']:.5f}")
            print(f"  - β=1.0: {maxFbetas[1.0]['max_fscore']:.5f}, precision: {maxFbetas[1.0]['precision']:.5f}, recall: {maxFbetas[1.0]['recall']:.5f}")
            print(f"IoU Scores:")
            print(f"  - Threshold 0.1: {IoU_multi_threshed[0.1]:.5f}")
            print(f"  - Threshold 0.3: {IoU_multi_threshed[0.3]:.5f}")
            print(f"  - Threshold 0.5: {IoU_multi_threshed[0.5]:.5f}")
            print(f"  - Threshold 0.7: {IoU_multi_threshed[0.7]:.5f}")
            
            tgt_img_path = meta_dict["tgt_image_path"][0]
            tgt_aolp_path = meta_dict["tgt_aolp_path"][0]
            tgt_dolp_path = meta_dict["tgt_dolp_path"][0]
            mask_path = meta_dict["tgt_mask_path"][0]
            supp_img_path = meta_dict["supp_image_path"][0]
            supp_aolp_path = meta_dict["supp_aolp_path"][0]
            supp_dolp_path = meta_dict["supp_dolp_path"][0]
            
            tgt_rgb_img = Image.open(tgt_img_path).convert('RGB')
            tgt_aolp_img = Image.open(tgt_aolp_path).convert('RGB')
            tgt_dolp_img = Image.open(tgt_dolp_path).convert('RGB')
            supp_rgt_img = Image.open(supp_img_path).convert('RGB')
            supp_aolp_img = Image.open(supp_aolp_path).convert('RGB')
            supp_dolp_img = Image.open(supp_dolp_path).convert('RGB')
            mask_img = Image.open(mask_path).convert('L')
            w, h = tgt_rgb_img.size
            
            resized_preds :dict = resize_predictions(pred, (h, w))
            
            thresholded_pred = (final_pred.cpu() > maxFbetas[0.3]['threshold']).float()
            thresholded_pred_resized = transforms.Resize((h, w))(thresholded_pred)
            
            thresholded_pred_img = (thresholded_pred_resized.numpy() * 255).astype(np.uint8)
            
            base_name = os.path.basename(os.path.abspath(os.path.join(tgt_img_path, "../../")))
            base_name = base_name + "_" + os.path.splitext(os.path.basename(tgt_img_path))[0]

            thresholded_output_path = os.path.join(dir_thresholded_outputs, f"{base_name}_thresholded_{maxFbetas[0.3]['max_fscore']:.3f}.png")
            Image.fromarray(thresholded_pred_img.squeeze()).save(thresholded_output_path)

            # サブプロットを作成
            num_imgs = 8 + len(resized_preds) + 1  # 8は最初の画像数、+1は最後の画像
            col = 4  # 列数
            row = (num_imgs + col - 1) // col  # 行数を計算
            fontsize_title = 12  # タイトルのフォントサイズ

            fig, axes = plt.subplots(row, col, figsize=(12, 16))  # figsizeを大きくして画像をはっきり表示
            fig.set_dpi(300)
            axes = axes.flatten()  # 1次元配列に変換

            # 最初の画像をプロット
            axes[0].imshow(tgt_rgb_img)
            axes[0].set_title("RGB Image", fontsize=fontsize_title)  # タイトルのフォントサイズを調整
            axes[0].axis('off')

            axes[1].imshow(tgt_aolp_img)
            axes[1].set_title("AoLP Image", fontsize=fontsize_title)
            axes[1].axis('off')

            axes[2].imshow(tgt_dolp_img)
            axes[2].set_title("DoLP Image", fontsize=fontsize_title)
            axes[2].axis('off')

            axes[3].imshow(mask_img, cmap='gray')
            axes[3].set_title("Ground Truth", fontsize=fontsize_title)
            axes[3].axis('off')

            axes[4].imshow(supp_rgt_img)
            axes[4].set_title("Supplementary RGB Image", fontsize=fontsize_title)
            axes[4].axis('off')

            axes[5].imshow(supp_aolp_img)
            axes[5].set_title("Supplementary AoLP Image", fontsize=fontsize_title)
            axes[5].axis('off')

            axes[6].imshow(supp_dolp_img)
            axes[6].set_title("Supplementary DoLP Image", fontsize=fontsize_title)
            axes[6].axis('off')

            start_idx = 7  
            for i, (key, _map) in enumerate(resized_preds.items()):
                idx = start_idx + i
                axes[idx].imshow(_map.squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[idx].set_title(key, fontsize=fontsize_title)
                axes[idx].axis('off')

            axes[start_idx + len(resized_preds)].imshow(thresholded_pred_resized.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[start_idx + len(resized_preds)].set_title("Thresholded Prediction", fontsize=fontsize_title)
            axes[start_idx + len(resized_preds)].axis('off')

            for ax in axes[start_idx + len(resized_preds) + 1:]:
                ax.axis('off')

            plt.suptitle(f"Max F-beta: {maxFbetas[0.3]['max_fscore']:.3f}, Thres: {maxFbetas[0.3]['threshold']:.3f}", fontsize=16)
            plt.tight_layout()  # padを増やして画像間のスペースを調整
            plt.subplots_adjust(top=0.92)  # タイトルとサブプロットが重ならないように
            visualization_output_file = os.path.join(dir_visualization_outputs, f"{base_name}_visualization_{maxFbetas[0.3]['max_fscore']:.3f}.png")
            plt.savefig(visualization_output_file, dpi=300)  # 高解像度で保存
            plt.close()
    
    avg_loss = np.mean(loss_list) if loss_list else 0.0
    avg_mae = np.mean(MAE_list) if MAE_list else 0.0
    avg_IoU :dict = {key: np.mean(values) if values else 0.0 for key, values in IoU_list.items()}
    avg_maxFbeta :dict  = {key: np.mean(values) if values else 0.0 for key, values in maxFbeta_list.items()}
    print(f"Average Loss: {avg_loss:.5f}")
    print(f"Average MAE: {avg_mae:.5f}")
    print("Average IoU Scores:")
    for key, value in avg_IoU.items():
        print(f"  - {key}: {value:.5f}")
    print("Average Max F-beta Scores:")
    for key, value in avg_maxFbeta.items():
        print(f"  - {key}: {value:.5f}")
    
    with open(results_txt_path, "w") as f:
        f.write(f"Average Test Loss: {avg_loss:.5f}\n")
        f.write(f"Average MAE: {avg_mae:.5f}\n")
        f.write("Average IoU Scores:\n")
        for key, value in avg_IoU.items():
            f.write(f"  - {key}: {value:.5f}\n")
        f.write("Average Max F-beta Scores:\n")
        for key, value in avg_maxFbeta.items():
            f.write(f"  - {key}: {value:.5f}\n")


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
    model.load_state_dict(torch.load(pram_path, weights_only=True), strict=False)
    model.eval()
    
    # Test the model
    if args.test_only:
        print("Test only mode")
        result_dir = os.path.join(args.result_dir, "test_outputs")
        print("Result directory: ", result_dir)
        test_model(model, test_loader, args, result_dir)
    else:
        for d_type, loader in data_type.items():
            # Create a directory to save the results
            print("Data type: ", d_type)
            result_dir = os.path.join(args.result_dir, d_type + "_outputs")
            print("Result directory: ", result_dir)
            test_model(model, loader, args, result_dir)


if __name__ == "__main__":
    main()
