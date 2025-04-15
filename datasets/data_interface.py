import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from .transforms import Transforms

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.batch_size = kwargs["batch_size"]
        self.image_size = kwargs["image_size"]
        self.num_workers = kwargs["num_workers"]
        self.aug_type = kwargs.get("aug_type", "default")
        self.train_dataset = None
        self.val_dataset = None
        self.base_transform =  Transforms.get_base_transform()  # 使用统一的基础转换
    
    def setup(self, stage=None):
        """加载并初始化数据集"""
        # 加载原始数据集（仅应用基础转换）
        full_dataset = datasets.ImageFolder(
            root=self.data_path,
            transform=self.base_transform
        )
        # 根据类型选择增强策略
        if self.aug_type == "light":
            train_transform = Transforms.get_light_augment_transform(self.image_size)
        elif self.aug_type == "strong":
            train_transform = Transforms.get_strong_augment_transform(self.image_size)
        else:  # 默认
            train_transform = Transforms.get_default_train_transform(self.image_size)
        val_test_transform = Transforms.get_val_test_transform(self.image_size)
        # 按类别划分索引
        class_indices = {}
        for idx, (_, label) in enumerate(full_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 对每个类别按比例分割 (60%训练，20%验证，20%测试)
        train_indices, val_indices, test_indices = [], [], []
        for label, indices in class_indices.items():
            np.random.seed(42)  # 固定随机种子
            np.random.shuffle(indices)
            n = len(indices)
            train_end = int(0.6 * n)
            val_end = train_end + int(0.2 * n)
            
            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])
            test_indices.extend(indices[val_end:])
        
        # 创建应用了不同transform的Subset
        self.train_dataset = ApplyTransform(
            Subset(full_dataset, train_indices),
            train_transform
        )
        self.val_dataset = ApplyTransform(
            Subset(full_dataset, val_indices),
            val_test_transform
        )
        self.test_dataset = ApplyTransform(
            Subset(full_dataset, test_indices),
            val_test_transform
        )

    def train_dataloader(self):
        """返回训练数据的 DataLoader"""
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)
    def val_dataloader(self):
        """返回验证数据的 DataLoader"""
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=True)
    def test_dataloader(self):
        """返回测试数据的 DataLoader"""
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=True)
        
class ApplyTransform(Subset):
    """动态应用Transform的Subset包装器"""
    def __init__(self, subset, transform=None):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, label