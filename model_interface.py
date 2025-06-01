import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from models import TransferNet


class MInterface(pl.LightningModule):
    """
    LightningModule 封装了你的 TransferNet（支持 MMD/LMMD/CORAL/ADV/DAAN/BNM）。
    - training_step 同时读取源域 batch 和 目标域 batch，计算 clf_loss + transfer_loss
    - validation_step/test_step 只在带标签的数据（源域验证 or 目标域测试）上做分类评估
    """

    def __init__(self, hparams):
        super().__init__()
        # 直接把所有 hparams 存进来，Lightning 会自动把它们存到 checkpoint 的 hyper_parameters 中
        self.save_hyperparameters(hparams)

        # 实例化纯 PyTorch 版的 TransferNet
        self.net = TransferNet(
            num_class=self.hparams.class_num,
            base_net=self.hparams.backbone,
            transfer_loss=self.hparams.transfer_loss,
            use_bottleneck=self.hparams.use_bottleneck,
            bottleneck_width=self.hparams.bottleneck_width,
            max_iter=self.hparams.max_iter
        )

        # 准备分类相关的指标
        metric_args = {
            "task": "multiclass",
            "num_classes": self.hparams.class_num,
            "average": "weighted"
        }
        # train 阶段，我们只记录源域分类准确率
        self.train_acc = Accuracy(**metric_args)

        # 验证/测试阶段记录 Accuracy / Precision / Recall / F1
        self.val_acc = Accuracy(**metric_args)
        self.val_precision = Precision(**metric_args)
        self.val_recall = Recall(**metric_args)
        self.val_f1 = F1Score(**metric_args)

        self.test_acc = Accuracy(**metric_args)
        self.test_precision = Precision(**metric_args)
        self.test_recall = Recall(**metric_args)
        self.test_f1 = F1Score(**metric_args)

        # 交叉熵用在验证与测试；训练时分类 loss 已在 TransferNet 内部计算
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # 纯前向用于 predict 接口，仅用于 test 阶段
        return self.net.predict(x)

    def training_step(self, batch, batch_idx):
        """
        batch['source'] = (x_s, y_s)
        batch['target'] = (x_t, y_t_dummy)
        """
        x_s, y_s = batch["source"]
        x_t, _   = batch["target"]

        # 让 TransferNet 同时跑 source/target，返回 (clf_loss, transfer_loss)
        clf_loss, transfer_loss = self.net(x_s, x_t, y_s)

        # 梯度回传时，TransferNet 内部会做 bottleneck / 对齐 等
        total_loss = clf_loss + self.hparams.transfer_loss_weight * transfer_loss

        # 记录日志
        self.log("train/clf_loss", clf_loss,    on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/transfer_loss", transfer_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # 只在源域上计算 train_acc
        with torch.no_grad():
            pred_s = torch.argmax(self.net.classifier_layer(
                self.net.bottleneck_layer(self.net.base_network(x_s))
                if self.hparams.use_bottleneck
                else self.net.base_network(x_s)
            ), dim=1)
            acc_s = self.train_acc(pred_s, y_s)
        self.log("train/acc_s", acc_s, on_step=True, on_epoch=True, prog_bar=True)

        # 如果使用 DAAN，TransferNet 会在 epoch 变化时通过 epoch_based_processing 更新动态参数
        self.net.epoch_based_processing(self.current_epoch, self.hparams.max_iter)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        验证阶段：只用源域验证集做分类评估
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.val_acc(preds, y)
        prec = self.val_precision(preds, y)
        rec = self.val_recall(preds, y)
        f1  = self.val_f1(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", prec, on_epoch=True)
        self.log("val_recall", rec, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        测试阶段：在目标域（如果指定）或源域验证集（如果未指定目标域）做分类评估
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.test_acc(preds, y)
        prec = self.test_precision(preds, y)
        rec = self.test_recall(preds, y)
        f1  = self.test_f1(preds, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=False)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_precision", prec, on_epoch=True)
        self.log("test_recall", rec, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        构造优化器和学习率调度器。这里沿用 SGD + LambdaLR，
        并按 TransferNet.get_parameters() 给不同子网分配不同 lr。
        """
        # 1) 取 parameter groups
        parameter_groups = self.net.get_parameters(initial_lr=self.hparams.lr)

        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )

        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                # inverse decay: lr * (1 + gamma * step)^{-decay}
                lr_lambda=lambda step: (1.0 + self.hparams.lr_gamma * float(step)) ** (-self.hparams.lr_decay)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "step"}
        else:
            return optimizer
