import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.batch_size = kwargs["batch_size"]
        self.image_size = kwargs["image_size"]
        self.num_workers = kwargs["num_workers"]
        self.train_dataset = None
        self.val_dataset = None
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def setup(self, stage=None):
        """加载并初始化数据集"""
        train_dataset = datasets.ImageFolder(root=self.data_path, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(root=self.data_path, transform=self.val_test_transform)
        test_dataset = datasets.ImageFolder(root=self.data_path, transform=self.val_test_transform)
        # 按类别划分索引
        class_indices = {}
        for idx, (_, label) in enumerate(train_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
    
        # 对每个类别按比例分割
        train_indices, val_indices, test_indices = [], [], []
        for label, indices in class_indices.items():
            np.random.seed(42)  # 固定随机种子以确保可重复性
            np.random.shuffle(indices)
            n = len(indices)
            train_end = int(0.6 * n)
            val_end = train_end + int(0.2 * n)
    
            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])
            test_indices.extend(indices[val_end:])
    
        # 创建 Subset
        self.train_subset = Subset(train_dataset, train_indices)
        self.val_subset = Subset(val_dataset, val_indices)
        self.test_subset = Subset(test_dataset, test_indices)

    def train_dataloader(self):
        """返回训练数据的 DataLoader"""
        return DataLoader(self.train_subset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)
    def val_dataloader(self):
        """返回验证数据的 DataLoader"""
        return DataLoader(self.val_subset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=True)
    def test_dataloader(self):
        """返回测试数据的 DataLoader"""
        return DataLoader(self.test_subset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=True)