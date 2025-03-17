import os
import math

import torch
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

def save_plots(train_loss: list, train_score: list, valid_loss: list, valid_score: list, save_path):
    """
    学習曲線を保存する

    :param train_loss: トレーニングデータの損失
    :param train_score: トレーニングデータのスコア
    :param valid_loss: バリデーションデータの損失
    :param valid_score: バリデーションデータのスコア
    :param save_path: 保存先のパス
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(range(len(train_loss)), train_loss, label="train_loss")
    ax[0].plot(range(len(valid_loss)), valid_loss, label="valid_loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(range(len(train_score)), train_score, label="train_score")
    ax[1].plot(range(len(valid_score)), valid_score, label="valid_score")
    ax[1].set_title("Score")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Score")
    ax[1].set_ylim(0, 1)  # Y軸の範囲を0から1に固定
    ax[1].legend()
    ax[1].grid(True)

    fig.suptitle("Training Curve, minimum loss epoch: {}, maximum score epoch: {}".format(
        valid_loss.index(min(valid_loss)), valid_score.index(max(valid_score))))
    plt.savefig(save_path)
    plt.close()


def plot_and_save(origin_image, origin_mask, origin_aolp, origin_dolp, pred, thres, epoch, maxFbeta, dir_val_outputs, file_name: str):
    num_graghs = len(pred)
    col = 4
    row = math.ceil((num_graghs + 5) / col)
    plt.figure(constrained_layout=True, figsize=(30, 25))
    
    plt.subplot(row, col, 1)
    plt.imshow(origin_image)
    plt.title("Image", fontsize=36)
    
    plt.subplot(row, col, 2)
    plt.imshow(origin_mask, cmap='gray')
    plt.title("GT", fontsize=36)
    
    plt.subplot(row, col, 3)
    plt.imshow(origin_aolp)
    plt.title("AoLP", fontsize=36)
    
    plt.subplot(row, col, 4)
    plt.imshow(origin_dolp)
    plt.title("DoLP", fontsize=36)

    for i in range(num_graghs):
        plt.subplot(row, col, i + 5)
        plt.imshow(pred[i], cmap='gray', vmin=0, vmax=1)
        title = ""
        if i == 0:
            title = "cnn based image"
        elif i == 1:
            title = "trans based image"
        elif i == 2:
            title = "cnn based aolp"
        elif i == 3:
            title = "trans based aolp"
        elif i == 4:
            title = "cnn based dolp"
        elif i == 5:
            title = "trans based dolp"
        elif i == 6:
            title = "GCG"
        elif i > 6:
            title = f"decoder {11 - i}"
        plt.title(title, fontsize=36)
        plt.colorbar()

    plt.subplot(row, col, num_graghs + 5)
    plt.imshow(pred[-1] > thres, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("max F_beta threshed", fontsize=36)

    plt.suptitle(f"epoch: {epoch}, maxFbeta: {maxFbeta:.3f}, thres: {thres:.3f}", fontsize=36)
    plt.savefig(os.path.join(dir_val_outputs, file_name))
    plt.close()
    
    
def particle_visualize(img: torch.Tensor, particle: torch.Tensor, save_path: str):
    """
    パーティクルの可視化を行う

    :param img: 画像 (3, H, W)
    :param particle: パーティクル (N, 2)
    :param save_path: 保存先のパス
    """
    img = img.permute(1, 2, 0).cpu().numpy()
    for i in range(particle.shape[0]):
        x, y = particle[i]
        img = cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)
    
    cv2.imwrite(save_path, img)


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb
    