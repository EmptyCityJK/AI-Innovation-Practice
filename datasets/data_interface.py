import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.batch_size = kwargs["batch_size"]
        self.image_size = kwargs["input_size"]
        self.num_workers = kwargs["num_workers"]
        self.train_dataset = None
        self.val_dataset = None
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def target_transform(label):
        return torch.eye(65)[label]
    def setup(self, stage=None):
        """加载并初始化数据集"""
        full_dataset = datasets.ImageFolder(root=self.data_path, 
                                            transform=self.transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

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