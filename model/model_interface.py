from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)
from .classifier import Model4Classifier

class MInterface(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  
        
        # 初始化指标
        num_classes = kwargs["class_num"]
        metric_args = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro"  # 可选'micro'/'weighted'
        }
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })
        
        self.val_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })
        
        self.test_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })

        self.model = Model4Classifier(**kwargs)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        
        metrics = getattr(self, f"{stage}_metrics")
        # 先记录loss和acc到进度条
        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=(stage != "test"), 
            on_step=False,
            on_epoch=True
        )
        self.log(
            f"{stage}_acc",
            metrics["acc"](y_pred, y),
            prog_bar=(stage != "test"),
            on_step=False,
            on_epoch=True
        )
        
        # 其他指标仅记录到日志不显示在进度条
        self.log_dict(
            {
                f"{stage}_precision": metrics["precision"](y_pred, y),
                f"{stage}_recall": metrics["recall"](y_pred, y),
                f"{stage}_f1": metrics["f1"](y_pred, y)
            },
            prog_bar=False,  # 关闭进度条显示
            on_step=False,
            on_epoch=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.epochs,
                eta_min=1e-5
            )
            return [optimizer], [scheduler]
        
        return optimizer