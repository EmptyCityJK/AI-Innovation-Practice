import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from .transforms import Transforms

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.batch_size = kwargs["batch_size"]
        self.image_size = kwargs["image_size"]
        self.num_workers = kwargs["num_workers"]
        self.k_fold = kwargs.get("k_fold", 0)
        self.current_fold = kwargs.get("current_fold", 0)
        self.aug_type = kwargs.get("aug_type", "default")
        
        # 根据类型选择增强策略
        if self.aug_type == "light":
            train_transform = Transforms.get_light_augment_transform(self.image_size)
        elif self.aug_type == "strong":
            train_transform = Transforms.get_strong_augment_transform(self.image_size)
        else:  # 默认
            train_transform = Transforms.get_default_train_transform(self.image_size)
        val_test_transform = Transforms.get_val_test_transform(self.image_size)

    def setup(self, stage=None):
        """加载并初始化数据集"""
        full_dataset = datasets.ImageFolder(root=self.data_path)
        if self.k_fold > 0:  # 交叉验证模式
            from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
            
            # 先划分20%作为独立测试集
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            all_idx = np.arange(len(full_dataset))
            labels = [label for _, label in full_dataset]
            train_val_idx, test_idx = next(sss.split(all_idx, labels))
            
            # K折划分训练验证集
            skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=42)
            splits = list(skf.split(train_val_idx, np.array(labels)[train_val_idx]))
            train_idx, val_idx = splits[self.current_fold]
            
            train_indices = train_val_idx[train_idx].tolist()
            val_indices = train_val_idx[val_idx].tolist()
            test_indices = test_idx.tolist()
        else:
            # 按类别划分索引
            class_indices = {}
            for idx, (_, label) in enumerate(full_dataset):
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
        self.train_subset = TransformDataset(Subset(full_dataset, train_indices), self.train_transform)
        self.val_subset = TransformDataset(Subset(full_dataset, val_indices), self.val_test_transform)
        self.test_subset = TransformDataset(Subset(full_dataset, test_indices), self.val_test_transform)

    def train_dataloader(self):
        """返回训练数据的 DataLoader"""
        return DataLoader(self.train_subset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          persistent_workers=True)

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