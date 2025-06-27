# 🎨 AI Innovation Practice: Transfer Learning on OfficeHome

这是一个为“人工智能创新实践”课程设计的期末项目。✨

本项目基于 **PyTorch** 和 **PyTorch Lightning** 框架，在 **Office-Home** 数据集上实现并复现了多种经典的无监督域自适应 (Unsupervised Domain Adaptation, UDA) 算法。

灵感和部分代码实现参考了优秀的迁移学习开源库 [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA)。

## 🌟 项目特性

- **模块化设计**: 基于 PyTorch Lightning 搭建，代码结构清晰，易于扩展和维护。
- **丰富的算法实现**: 实现了多种主流的域自适应方法，涵盖了不同的技术路线。
- **灵活的配置**: 所有实验超参数均可通过 `.yaml` 文件进行配置，方便调参和复现。
- **强大的数据增强**: 内置了多种数据增强策略，可灵活选择。

## 🚀 已实现的算法

项目中包含了以下经典的 UDA 算法：

| 算法                 | 核心思想                                           | 配置文件                   | 参考文献                                                     |
| -------------------- | -------------------------------------------------- | -------------------------- | ------------------------------------------------------------ |
| 🤖 **DAN**            | 多核最大均值差异 (Multi-Kernel MMD)                | `DAN/DAN.yaml`             | [Learning Transferable Features with Deep Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) |
| adversarial **DANN** | 域对抗训练 (Domain-Adversarial Training)           | `DANN/DANN.yaml`           | [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495) |
| 🐠 **DeepCoral**      | 最小化源域与目标域的二阶统计量（协方差）差异       | `DeepCoral/DeepCoral.yaml` | [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf) |
| 🌐 **DSAN**           | 局部最大均值差异 (Local MMD)，对齐子域名           | `DSAN/DSAN.yaml`           | [Deep Subdomain Adaptation Network for Image Classification](https://arxiv.org/abs/1905.10953) |
| ⚖️ **DAAN**           | 动态对抗适应网络 (Dynamic Adversarial Adaptation)  | `DAAN/DAAN.yaml`           | [Transfer Learning with Dynamic Adversarial Adaptation Network](https://ieeexplore.ieee.org/abstract/document/8970703) |
| ⚛️ **BNM**            | 批量核范数最大化 (Batch Nuclear-norm Maximization) | `BNM/BNM.yaml`             | [Towards Discriminability and Diversity: Batch Nuclear-norm Maximization](http://arxiv.org/abs/2003.12237) |

## 📁 项目结构

```
.
├── main.py                # 🚀 主训练脚本
├── model_interface.py     # ⚡️ PyTorch Lightning 核心模块
├── models.py              # 🧠 迁移网络结构 TransferNet
├── backbones.py           # 🦴 特征提取骨干网络 (e.g., ResNet50)
├── datasets/
│   ├── data_interface.py  # 📦 数据加载模块
│   └── transforms.py      # ✨ 数据增强策略
├── loss_funcs/            # 📉 各种对齐损失函数的实现
│   ├── mmd.py
│   ├── coral.py
│   └── ...
├── transfer_losses.py     # 🎁 统一的迁移损失接口
├── checkpoints/           # 💾 保存训练好的模型
├── logs/                  # 📊 保存 TensorBoard 和 Wandb 日志
└── DAN/                   # 每个算法一个文件夹
    ├── DAN.yaml           # ⚙️ 算法配置文件
    └── DAN.sh             # 📜 运行脚本示例
```

## 🛠️ 环境准备

1. **克隆仓库**

   Bash

   ```
   git clone https://github.com/emptycityjk/ai-innovation-practice.git
   cd ai-innovation-practice
   ```

2. **创建 Conda 环境 (推荐)**

   Bash

   ```
   conda create -n uda python=3.8
   conda activate uda
   ```

3. 安装依赖

   本项目主要依赖 PyTorch, PyTorch Lightning 等。你可以通过 pip 安装它们：

   Bash

   ```
   # 根据你的 CUDA 版本选择合适的 PyTorch 安装命令
   # 访问 https://pytorch.org/get-started/locally/
   pip install torch torchvision torchaudio
   
   pip install pytorch-lightning configargparse torchmetrics
   ```

4. 准备数据集

   请从官网下载 Office-Home Dataset 并解压。然后在 main.py 或 .sh 脚本中指定数据集的根目录。数据集目录结构应如下所示：

   ```
   /path/to/your/datasets/
   └── OfficeHome/
       ├── Art/
       ├── Clipart/
       ├── Product/
       └── RealWorld/
   ```

## 🎮 如何运行

你可以通过执行 `main.py` 并指定配置文件和数据路径来运行任何一个算法的实验。所有的算法脚本都提供了示例。

以 **DAN** 算法为例，将源域 **Art (A)** 迁移到目标域 **Clipart (C)**：

1. **修改配置 (可选)**: 打开 `DAN/DAN.yaml` 文件，你可以按需调整学习率、批大小等超参数。

2. **修改脚本 (推荐)**: 打开 `DAN/DAN.sh` 文件，将 `data_dir` 修改为你的 OfficeHome 数据集路径。

3. **开始训练!**

   Bash

   ```
   bash DAN/DAN.sh
   ```

   或者直接使用 `python` 命令：

   Bash

   ```
   python main.py --config DAN/DAN.yaml \
                  --data_path /path/to/your/datasets/OfficeHome \
                  --src_domain Art \
                  --tgt_domain Clipart
   ```

日志和模型权重将自动保存在 `logs/` 和 `checkpoints/` 目录下。
