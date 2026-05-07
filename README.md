DFS-YOLO: Dynamic Multi-Scale Perception and Contextual Fusion for Small Object Detection in UAV Imagery
# DFS-YOLO: Enhanced YOLO for UAV Small Object Detection 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange.svg)
![License](https://img.shields.io/badge/License-AGPL%203.0-green.svg)

## 📖 简介 (Introduction)

DFS-YOLO 是一个基于 Ultralytics YOLO 架构进行深度重构与优化的先进目标检测模型，专门针对**无人机（UAV）航拍视角**和**极小目标检测（Tiny Object Detection）**场景设计。

在无人机视角下，目标通常面临尺度剧烈变化、背景复杂以及分辨率极低等挑战。为了解决这些问题，本项目在底层架构中引入了动态多尺度卷积、细粒度注意力切片融合以及全维度特征提取机制，显著提升了模型对微小特征的感知能力。

## ✨ 核心创新点 (Key Features)

本项目对 YOLO 的网络结构（Backbone & Head）进行了大幅魔改，主要集成了以下核心模块：

* **DMB (Dynamic Multi-scale Block) & C3k2_DMB**: 
  重构自 Kernel Warehouse (KW) 机制。针对无人机高度变化带来的尺度问题，引入了共享权重的多尺度膨胀卷积（Dilations=[1, 2]），并结合了定制的 `UAV_Attention`（双路特征池化），在保留背景上下文的同时，极大地增强了对极小目标的局部高亮特征提取。
* **SCA_Fusion (Spatial-Channel Attention Fusion)**: 
  一种针对小目标优化的多级融合网络。内部集成了细粒度网格切片无参数注意力（SimAM with Slicing）、双池化通道注意力（捕捉全局背景与小目标显著性）以及空洞空间注意力，实现像素级的精细特征融合。
* **CSPOKnet (CSP OmniKernel Network)**: 
  引入全维度大感受野卷积机制，通过 CSP（跨阶段局部网络）结构封装，在保证推理速度的同时，有效捕捉大范围的复杂背景特征。
* **SPDConv (Space-to-Depth Convolution)**: 
  取代传统步长卷积，通过空间到深度的转换，最大程度避免了下采样过程中细粒度特征的丢失，是小目标检测的利器。

## 🛠️ 安装指南 (Installation)

建议使用 Conda 创建独立的虚拟环境来运行本项目。

```bash
# 1. 克隆仓库
git clone [https://github.com/your_username/DFS-YOLO.git](https://github.com/your_username/DFS-YOLO.git)
cd DFS-YOLO

# 2. 创建并激活虚拟环境 (推荐 Python 3.10)
conda create -n dfs-yolo python=3.10
conda activate dfs-yolo

# 3. 安装 PyTorch (请根据你的 CUDA 版本调整)
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装依赖
pip install -r requirements.txt
