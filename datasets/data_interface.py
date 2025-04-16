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

class MixupCutmixTransform:
    def __init__(self, alpha=1.0, cutmix_prob=0.5, num_classes=65):
        """
        alpha: mixup/cutmix的分布参数
        cutmix_prob: 执行cutmix的概率（剩余概率执行mixup）
        """
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes
        self.lambda_ = None
        self.index = None

    def _rand_bbox(self, size, lam):
        """生成cutmix的边界框"""
        W, H = size[2], size[3]
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - int(W * np.sqrt(1 - lam)) // 2, 0, W)
        bby1 = np.clip(cy - int(H * np.sqrt(1 - lam)) // 2, 0, H)
        bbx2 = np.clip(cx + int(W * np.sqrt(1 - lam)) // 2, 0, W)
        bby2 = np.clip(cy + int(H * np.sqrt(1 - lam)) // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        images, labels = batch  # 输入为整个batch的数据
        batch_size = images.size(0)
        
        # 生成混合参数
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # 生成one-hot标签
        labels_onehot = torch.zeros(
            batch_size, self.num_classes, 
            device=images.device
        ).scatter_(1, labels.unsqueeze(1), 1)
        
        # 随机选择mixup或cutmix
        if np.random.rand() < self.cutmix_prob:
            # CutMix
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            final_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        else:
            # Mixup
            final_lam = lam
            images = images * final_lam + images[rand_index] * (1 - final_lam)
        
        # 混合标签
        mixed_labels = labels_onehot * final_lam + labels_onehot[rand_index] * (1 - final_lam)
        
        return images, mixed_labels

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
            self.train_transform = Transforms.get_light_augment_transform(self.image_size)
        elif self.aug_type == "strong":
            self.train_transform = Transforms.get_strong_augment_transform(self.image_size)
        else:  # 默认
            self.train_transform = Transforms.get_default_train_transform(self.image_size)
        self.val_test_transform = Transforms.get_val_test_transform(self.image_size)
        # 新增参数
        self.mixup_alpha = kwargs.get("mixup_alpha", 0.0)  # 0表示禁用
        self.cutmix_prob = kwargs.get("cutmix_prob", 0.5)

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
                          # collate_fn=self._mixup_collate,
                          persistent_workers=True,
                          )
    def _mixup_collate(self, batch):
        # 原始数据加载
        images = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        
        # 应用mixup/cutmix
        if self.mixup_alpha > 0 and self.training:  # 仅在训练时启用
            transform = MixupCutmixTransform(
                alpha=self.mixup_alpha,
                cutmix_prob=self.cutmix_prob,
                num_classes=self.num_classes  # 需要添加num_classes参数到数据模块
            )
            images, labels = transform((images, labels))
            
        return images, labels
                          
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