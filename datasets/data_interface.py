import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
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
        full_dataset = datasets.ImageFolder(
            root=self.data_path, 
            transform=self.base_transform
        )
        # 划分训练(70%)/验证(15%)/测试(15%)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_subset, val_subset, test_subset = random_split(
            full_dataset, 
            [train_size, val_size, test_size]
        )
        
        # 根据类型选择增强策略
        if self.aug_type == "light":
            train_transform = Transforms.get_light_augment_transform(self.image_size)
        elif self.aug_type == "strong":
            train_transform = Transforms.get_strong_augment_transform(self.image_size)
        else:  # 默认
            train_transform = Transforms.get_default_train_transform(self.image_size)
            
        # 应用转换
        self.train_dataset = ApplyTransform(train_subset, train_transform)
        self.val_dataset = ApplyTransform(val_subset, Transforms.get_val_test_transform(self.image_size))
        self.test_dataset = ApplyTransform(test_subset, Transforms.get_val_test_transform(self.image_size))

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