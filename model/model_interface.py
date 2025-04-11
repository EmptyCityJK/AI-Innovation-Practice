from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torchmetrics
from torchmetrics.classification import Accuracy
from .model import SimpleFCN

class MInterface(LightningModule):
    def __init__(self, **kwargs):
        super(MInterface, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = kwargs["learning_rate"]
        self.model = SimpleFCN(**kwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        acc = Accuracy(y_hat.argmax(dim=1), y)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        acc = Accuracy(y_hat.argmax(dim=1), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
