import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics import Accuracy

# 数据模块
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, img_size=(64, 64)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    
    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # 划分数据集：80%训练，10%验证，10%测试
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

# 全连接网络模型
class SimpleFCN(pl.LightningModule):
    def __init__(self, input_size=64 * 64 * 3, hidden_size=512, num_classes=65):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, num_classes)
        )
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == '__main__':
    # 初始化数据模块和模型
    datamodule = ImageDataModule(data_dir='./data', img_size=(64, 64))
    model = SimpleFCN(input_size=64 * 64 * 3)
    
    # 训练配置
    trainer = Trainer(
        max_epochs=20,
        accelerator='auto',
        devices='auto',
        logger=True,
        enable_progress_bar=True
    )
    
    # 训练和测试
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)