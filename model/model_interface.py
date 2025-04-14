from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torchmetrics
from torchmetrics.classification import Accuracy
from .model import SimpleFCN
from .classifier import Model4Classifier

class MInterface(LightningModule):
    def __init__(self, **kwargs):
        super(MInterface, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = kwargs["learning_rate"]
        self.accuracy = Accuracy(
            task='multiclass', 
            num_classes=kwargs["class_num"]
        )
        self.model = Model4Classifier(**kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        acc = self.accuracy(y_hat, y)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
