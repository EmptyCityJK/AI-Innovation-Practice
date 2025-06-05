import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from itertools import cycle
from models import TransferNet


class MInterface(pl.LightningModule):
    """
    LightningModule 封装 PyTorch 版的 TransferNet，内部使用“无限迭代器”来实现应用 InfiniteDataLoader 的效果。
    - on_train_start(): 初始化 source_iter 和 target_iter，都用 itertools.cycle()
    - training_step(): 忽略 Lightning 传入的 batch，直接从 source_iter / target_iter 中获取下一批数据
    - validation_step/test_step(): 在目标域带标签数据上做分类评估，与原生 PyTorch 版一致
    """

    def __init__(self, hparams):
        super().__init__()
        # 保存所有超参数（包括 max_iter、n_iter_per_epoch、epoch_based_training 等）
        self.save_hyperparameters(hparams)

        # 实例化原生 PyTorch 版的 TransferNet
        self.net = TransferNet(
            num_class=self.hparams.class_num,
            base_net=self.hparams.backbone,
            transfer_loss=self.hparams.transfer_loss,
            use_bottleneck=self.hparams.use_bottleneck,
            bottleneck_width=self.hparams.bottleneck_width,
            max_iter=self.hparams.max_iter
        )

        # 构造分类指标，和原版保持一致
        metric_args = {
            "task": "multiclass",
            "num_classes": self.hparams.class_num,
            "average": "weighted"
        }
        self.train_metrics = nn.ModuleDict({
            "micro_acc": Accuracy(
                task="multiclass",
                num_classes=self.hparams.class_num,
                average="micro"
            ),
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })

        self.val_metrics = nn.ModuleDict({
            "micro_acc": Accuracy(
                task="multiclass",
                num_classes=self.hparams.class_num,
                average="micro"
            ),
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })

        self.test_metrics = nn.ModuleDict({
            "micro_acc": Accuracy(
                task="multiclass",
                num_classes=self.hparams.class_num,
                average="micro"
            ),
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })

        # 验证/测试阶段使用的交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # 在 on_train_start 里我们会创建这两个无限迭代器
        self.source_iter = None
        self.target_iter = None

    def forward(self, x):
        # 纯前向，仅用于 test 阶段调用 predict()
        return self.net.predict(x)

    def on_train_start(self):
        """
        在训练一开始时构造“无限 iter”：
        - self.source_iter: 对 source_train_dataloader 应用 itertools.cycle
        - self.target_iter: 对 target_train_dataloader 应用 itertools.cycle
        这样在 training_step 里我们就可以不断 next()，永不 StopIteration。
        """
        # 拿到 DataModule 提供的两个 DataLoader
        source_loader = self.trainer.datamodule.source_train_dataloader()
        target_loader = self.trainer.datamodule.target_train_dataloader()

        # 如果 target_loader=None（无目标域情况），就不做 cycle
        self.source_iter = cycle(source_loader)
        if target_loader is not None:
            self.target_iter = cycle(target_loader)
        else:
            self.target_iter = None

    def training_step(self, batch, batch_idx):
        """
        忽略 Lightning 传入的 batch，直接用无限迭代器 source_iter/target_iter：
        - x_s, y_s 从 source_iter 拿
        - x_t, _   从 target_iter 拿（若 target_iter 为空，则 x_t=None，不做对齐）
        然后与原生 PyTorch 版保持一致地计算 clf_loss & transfer_loss，并更新网络。
        """
        # 1. 从无限迭代器中取下一批源域样本
        x_s, y_s = next(self.source_iter)
        x_s, y_s = x_s.to(self.device), y_s.to(self.device)

        # 2. 从无限迭代器中取下一批目标域样本（如果有）
        if self.target_iter is not None:
            x_t, _ = next(self.target_iter)
            x_t = x_t.to(self.device)
        else:
            x_t = None

        # 3. 让 TransferNet 同时计算 source/target，或仅计算 source(若 x_t=None)
        clf_loss, transfer_loss = self.net(x_s, x_t, y_s) if x_t is not None else (self.net(x_s, x_s, y_s)[0], 0.0)
        total_loss = clf_loss + self.hparams.transfer_loss_weight * transfer_loss

        # 4. 记录 loss
        self.log("train_clf_loss", clf_loss,    on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_transfer_loss", transfer_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=False)

        # 5. 只在源域上计算 train_acc（和 PyTorch 版保持一致）
        with torch.no_grad():
            features = self.net.base_network(x_s)
            if self.hparams.use_bottleneck:
                features = self.net.bottleneck_layer(features)
            pred_s = torch.argmax(self.net.classifier_layer(features), dim=1)
            acc = self.train_metrics["acc"](pred_s, y_s)
            micro_acc = self.train_metrics["micro_acc"](pred_s, y_s)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_micro_acc", micro_acc, on_step=True, on_epoch=False, prog_bar=False)

        # 6. 如果使用 DAAN，则在 epoch 开始时更新动态权重
        self.net.epoch_based_processing(self.current_epoch, self.hparams.max_iter)

        # 7. 记录当前 step 的 lr（取 param_groups[0]）
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        验证阶段：只在目标域带标签数据上做分类评估，与 PyTorch 版在 train() 后的 test() 等效。
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.val_metrics["acc"](preds, y)
        micro_acc = self.val_metrics["micro_acc"](preds, y)
        prec = self.val_metrics["precision"](preds, y)
        rec = self.val_metrics["recall"](preds, y)
        f1  = self.val_metrics["f1"](preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_micro_acc", micro_acc, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", prec, on_epoch=True, prog_bar=False)
        self.log("val_recall", rec, on_epoch=True, prog_bar=False)
        self.log("val_f1", f1, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        """
        测试阶段：同验证阶段，在目标域带标签数据上做评估。
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.test_metrics["acc"](preds, y)
        micro_acc = self.test_metrics["micro_acc"](preds, y)
        prec = self.test_metrics["precision"](preds, y)
        rec = self.test_metrics["recall"](preds, y)
        f1  = self.test_metrics["f1"](preds, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=False)
        self.log("test_micro_acc", micro_acc, on_epoch=True, prog_bar=False)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_precision", prec, on_epoch=True, prog_bar=False)
        self.log("test_recall", rec, on_epoch=True, prog_bar=False)
        self.log("test_f1", f1, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        """
        - initial_lr_for_groups = 1.0 if lr_scheduler=True else args.lr
        - optimizer = SGD(param_groups, lr=args.lr, momentum, weight_decay)
        - scheduler（如果启用） = LambdaLR(step 衰减)：lr * (1 + γ·step)^(-decay)
        """
        initial_lr_for_groups = self.hparams.lr

        # 2) 构造参数组
        parameter_groups = self.net.get_parameters(initial_lr=initial_lr_for_groups)

        # 3) 构造 SGD
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )

        # 4) 如果启用调度器，则每 step 做一次 LambdaLR
        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: (1.0 + self.hparams.lr_gamma * float(step)) ** (-self.hparams.lr_decay)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "interval": "step"
            }
        else:
            return optimizer
