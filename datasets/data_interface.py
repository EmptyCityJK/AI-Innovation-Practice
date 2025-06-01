# datasets/data_interface.py

import os
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from .transforms import Transforms

class TransformDataset(Dataset):
    """
    将 Subset 包一层，好方便在 __getitem__ 时给 PIL Image 贴上 transform。
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]   # x: PIL Image, y: int
        if self.transform:
            x = self.transform(x)   # 将 PIL Image 转为 Tensor
        return x, y

    def __len__(self):
        return len(self.subset)


class DInterface(pl.LightningDataModule):
    """
    支持领域自适应的 DataModule。
    - source_domain: 源域目录名（如 "RealWorld"）
    - target_domain: 目标域目录名（如 "Clipart" 或者 None。如果为 None，则退化为普通迁移学习，只在源域验证/测试）
    - val_ratio: 源域 train/val 拆分比例
    """

    def __init__(
        self,
        data_path: str,
        source_domain: str = "RealWorld",
        target_domain: str = None,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        aug_type: str = "default",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.aug_type = aug_type
        self.val_ratio = val_ratio
        self.seed = seed

        # OfficeHome 固定 65 类
        self.num_classes = 65

        # 根据 aug_type 选择训练阶段的 transform
        if self.aug_type == "light":
            self.train_transform = Transforms.get_light_augment_transform(self.image_size)
        elif self.aug_type == "strong":
            self.train_transform = Transforms.get_strong_augment_transform(self.image_size)
        else:
            self.train_transform = Transforms.get_default_train_transform(self.image_size)

        # 验证/测试阶段统一使用标准化 transform
        self.val_test_transform = Transforms.get_val_test_transform(self.image_size)

        # 以下属性会在 setup() 中被赋值
        self.source_train_dataset = None
        self.source_val_dataset = None
        self.target_train_dataset = None
        self.target_test_dataset = None

    def prepare_data(self):
        # OfficeHome 数据集一般手动下载，此处无需自动下载
        pass

    def setup(self, stage=None):
        """
        数据拆分逻辑：
        1. 源域: 按类别 stratified 拆分 train/val
        2. 目标域: 全量 ImageFolder 作为无标签 train（用于对齐）和 test（如果带标签）
        """
        # —— Step 1: 源域 ImageFolder (transform=None) —— #
        source_dir = os.path.join(self.data_path, self.source_domain)
        assert os.path.isdir(source_dir), f"源域目录 '{source_dir}' 不存在"
        full_source = datasets.ImageFolder(root=source_dir, transform=None)
        # 按类别做 stratified 拆分
        labels = [lab for _, lab in full_source]
        class_indices = {}
        for idx, lab in enumerate(labels):
            class_indices.setdefault(lab, []).append(idx)

        train_indices = []
        val_indices = []
        np.random.seed(self.seed)
        for lab, idx_list in class_indices.items():
            np.random.shuffle(idx_list)
            n = len(idx_list)
            n_val = int(self.val_ratio * n)
            val_indices.extend(idx_list[:n_val])
            train_indices.extend(idx_list[n_val:])

        source_train_full = Subset(full_source, train_indices)
        source_val_full   = Subset(full_source, val_indices)

        self.source_train_dataset = TransformDataset(source_train_full, transform=self.train_transform)
        self.source_val_dataset   = TransformDataset(source_val_full,   transform=self.val_test_transform)

        # —— Step 2: 目标域无标签 train，和带标签 test —— #
        if self.target_domain is not None:
            target_dir = os.path.join(self.data_path, self.target_domain)
            assert os.path.isdir(target_dir), f"目标域目录 '{target_dir}' 不存在"
            full_target = datasets.ImageFolder(root=target_dir, transform=None)
            # “目标无标签” train dataset
            self.target_train_dataset = TransformDataset(full_target, transform=self.train_transform)
            # “目标带标签” test dataset
            self.target_test_dataset  = TransformDataset(full_target, transform=self.val_test_transform)
        else:
            # 如果不指定目标域，则不进行对齐，退化为源域验证/测试
            self.target_train_dataset = None
            self.target_test_dataset  = None

    def train_dataloader(self):
        """
        Lightning 会自动将字典形式的多个 loader 组合成 CombinedLoader，
        在 training_step(self, batch, batch_idx) 中，batch 会是一个 dict：
            batch = {'source': (x_s, y_s), 'target': (x_t, y_t_dummy)}
        如果 target_loader=None，Lightning 会忽略它，仅返回 source_loader
        """
        source_loader = DataLoader(
            self.source_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )

        if self.target_train_dataset is not None:
            target_loader = DataLoader(
                self.target_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True
            )
        else:
            target_loader = None

        # Lightning 会把 {'source': source_loader, 'target': target_loader} 变成 CombinedLoader
        return {"source": source_loader, "target": target_loader}

    def val_dataloader(self):
        # 验证时仅在源域做分类验证
        return DataLoader(
            self.source_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        # 测试时如果指定了 target_domain，则在目标域全量做测试；否则复用源域验证集
        if self.target_test_dataset is not None:
            return DataLoader(
                self.target_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.source_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
