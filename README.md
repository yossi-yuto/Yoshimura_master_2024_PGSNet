# 吉村（修論：RGB・偏光版）の引き継ぎコード
独自の鏡面物体データセットにおけるPGSNetと拡張版（提案法）の比較実験のためのコード．

## Requirement
- Python 3.12.3
- pytorch 2.5.0

## 鏡面物体データセット
鏡面物体データセットは，5種類の異なる形状の鏡面を撮影した，RGB画像と，各RGBチャンネルに対応する偏光画像で構成される．本研究では，異なる4方向の偏光素子を内蔵したカラー偏光カメラ[1]を用い，2つの視点から対象の鏡面を撮影した．正解マスク画像は2視点画像ペアのうち，検出対象である視点画像（ターゲット画像）に対してアノテーションが付与されている．本実験では，5種類の鏡面のうち，4種類を学習用，残りの1種類をテストデータとして，PGSNet[2]と提案法を評価した．

- [OneDrive Drive link](https://drive.google.com/drive/folders/1-Z1_ngBneZgS6wBlmV8amAZwiPJiXn56?usp=sharing)
- データセットの内訳

    | 鏡面の種類<br>(フォルダ名)           | 平面鏡<br>(planes) | トレイ<br>(tray) | コップ<br>(cup) | 凸面鏡<br>(sph) | 灰皿<br>（ash） | 合計 |
    |------------|-------|-------|--------|--------|------|----|
    | データ数   | 10    | 10    | 11     | 11     | 12   | 54 |


## Data structure
- Google driveにアップロードされたzipファイル（```videos.zip```）を```./videos```ダウンロードし全て展開する


```bash
.videos/
├── tray/                             # 鏡面の種類（トレイ）
│   ├── 20241108_211146_fps_5.0/      # ビデオID（撮影日時_FPS）
│   │   ├── AoLP/                     # 偏光角画像（.tiff）
│   │   ├── DoLP/                     # 偏光度画像（.tiff）
│   │   ├── RGB/                      # RGBフレーム画像（.jpg）
│   │   ├── Visual/                   # ヒートマップ可視化画像（.jpg）
│   │   ├── annotation/               # アノテーションファイル（.json, .txt）
│   │   └── mask/                     # 鏡面領域のマスク画像（.png）
│   └── 20241108_211330_fps_5.0/
│       └── ...
│
└── cup/                              # 鏡面の種類（コップ）
    ├── 20241111_152100_fps_5.0/
    │   ├── AoLP/
    │   ├── DoLP/
    │   ├── RGB/
    │   ├── Visual/
    │   ├── annotation/
    │   └── mask/
    └── ...
```

<!-- ./videos
├── tray #鏡面の種類
│   ├── 20241108_211146_fps_5.0 #ビデオID
    │   ├── AoLP # 偏光角画像を格納
    │   │   ├── 0000_aolp.tiff
    │   │   ├── 0001_aolp.tiff
    │   │   ├── 0002_aolp.tiff
    │   │   └── 0003_aolp.tiff
    │   ├── DoLP # 偏光度画像を格納
    │   │   ├── 0000_dolp.tiff
    │   │   ├── 0001_dolp.tiff
    │   │   ├── 0002_dolp.tiff
    │   │   └── 0003_dolp.tiff
    │   ├── RGB # RGBフレームを格納
    │   │   ├── 0000_rgb.jpg
    │   │   ├── 0001_rgb.jpg
    │   │   ├── 0002_rgb.jpg
    │   │   └── 0003_rgb.jpg
    │   ├── Visual # 数値を可視化したヒートマップを格納
    │   │   ├── 0000_aolp_crop.jpg
    │   │   ├── 0000_dolp_crop.jpg
    │   │   ├── 0001_aolp_crop.jpg
    │   │   ├── 0001_dolp_crop.jpg
    │   │   ├── 0002_aolp_crop.jpg
    │   │   ├── 0002_dolp_crop.jpg
    │   │   ├── 0003_aolp_crop.jpg
    │   │   └── 0003_dolp_crop.jpg
    │   ├── anotation # ターゲット画像とソース画像の設定が格納
    │   │   ├── 0001_rgb.json # ターゲット画像のIDを格納
    │   │   └── first_frame.txt # ソース画像のIDを格納
    │   └── mask # ターゲット画像に対応した鏡面領域のマスク画像を格納
    │       └── 0001_rgb.png 
    ├── 20241108_211330_fps_5.0
    │   ├──...
    │ 
    └──...

└── cup
    ├── 20241111_152100_fps_5.0
        ├── AoLP
        │   ├── 0000_aolp.tiff
        │   ├── 0001_aolp.tiff
        │   ├── 0002_aolp.tiff
        │   └── 0003_aolp.tiff
        ├── DoLP
        │   ├── 0000_dolp.tiff
        │   ├── 0001_dolp.tiff
        │   ├── 0002_dolp.tiff
        │   └── 0003_dolp.tiff
        ├── RGB
        │   ├── 0000_rgb.jpg
        │   ├── 0001_rgb.jpg
        │   ├── 0002_rgb.jpg
        │   └── 0003_rgb.jpg
        ├── Visual
        │   ├── 0000_aolp_crop.jpg
        │   ├── 0000_dolp_crop.jpg
        │   ├── 0001_aolp_crop.jpg
        │   ├── 0001_dolp_crop.jpg
        │   ├── 0002_aolp_crop.jpg
        │   ├── 0002_dolp_crop.jpg
        │   ├── 0003_aolp_crop.jpg
        │   └── 0003_dolp_crop.jpg
        ├── anotation
        │   ├── 0001_rgb.json
        │   └── first_frame.txt
        └── mask
            └── 0001_rgb.png -->


## Installation
実験は以下の手順により再現
1. リポジトリーのclone
```bash
$ git clone https://github.com/yossi-yuto/Yoshimura_master_2024_PGSNet.git
```
2.  ```requirements.txt``` に記載されたパーケージのインストール
```bash
$ pip install -r requirements.txt
```
3. Conformerの事前学習済みパラメータ```Conformer_base_patch16.pth```を[OneDrive](https://wakayamauniv-my.sharepoint.com/:u:/g/personal/s246316_wakayama-u_ac_jp/ETbSjl7rSx1DnDI7F3tb6GgBQpP24xinZgTMnlSY16icGQ?e=kpywQj)からダウンロードし，```./pretrain_param```ディレクトリに保存

## How to run
提案法のモデルを5foldの交差検証で実施する場合，以下のように実行．
```bash 
$ source train_infer.sh {GPU_NUM} {date}
```
`{GPU_NUM}`はGPUのデバイスを指定し，`{date}`は実行日時を記載．

### Example
GPUデバイスの０番を使用し、2025年2月10日に実行する場合、以下のコマンドを実行.

```bash
$ source train_infer.sh 0 20250210
```
なお、実験結果は```./results```に格納
```
./results
└── 20250210_seed42
    ├── PGS # PGSNetの実験結果
    └── proposed # 拡張版の実験結果
```

### Reference
[1] LUCID Vision Labs, I.: Triton 5 MP Polarization Camera, https://thinklucid.com/ja/product/triton-5-mp-polarization-camera/. Accessed: 2024-03-14.

[2] Haiyang Mei, Bo Dong, Wen Dong, Jiaxi Yang, Seung-Hwan Baek, Felix Heide, Pieter Peers, Xiaopeng Wei, Xin Yang, "Glass Segmentation using Intensity and Spectral Polarization Cues," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022.
[project page](https://mhaiyang.github.io/CVPR2022_PGSNet)