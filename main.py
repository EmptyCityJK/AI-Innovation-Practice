import configargparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datasets.data_interface import DInterface
from model_interface import MInterface

def train(args):
    os.environ["WANDB_MODE"] = "offline"
    # 让 cuDNN 挑最优 Conv 算法
    torch.backends.cudnn.benchmark = True

    # 1. Logger (TensorBoard + Wandb)
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.project_name,
        version=None
    )
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=f"{args.config}{args.src_domain}_to_{args.tgt_domain}" if args.tgt_domain else f"{args.src_domain}_only",
        save_dir=args.log_dir,
        config=args,
    )

    # 2. DataModule：把 epoch_based_training 传进去
    data_module = DInterface(
        data_path=args.data_path,
        source_domain=args.src_domain,
        target_domain=args.tgt_domain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        aug_type=args.aug_type,
        seed=args.seed,
        epoch_based=args.epoch_based_training
    )
    data_module.setup()

    # 3. 计算 max_iter (给 DAAN 用)
    source_train_size = len(data_module.source_train_dataset)
    target_train_size = len(data_module.target_train_dataset) if data_module.target_train_dataset is not None else 0
    len_source_loader = source_train_size // args.batch_size
    len_target_loader = target_train_size // args.batch_size

    if args.epoch_based_training:
        iter_per_epoch = min(len_source_loader, len_target_loader)
    else:
        iter_per_epoch = args.n_iter_per_epoch

    if iter_per_epoch == 0:
        iter_per_epoch = args.n_iter_per_epoch

    args.max_iter = args.n_epoch * iter_per_epoch
    print(f"len(source_loader)={len_source_loader}, len(target_loader)={len_target_loader}, iter_per_epoch={iter_per_epoch}, max_iter={args.max_iter}")

    # 4. LightningModule
    hparams = vars(args)
    model = MInterface(hparams)

    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
        dirpath=args.checkpoint_dir,
        filename="best-{epoch:02d}-{val_acc:.4f}"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=args.early_stop,
        mode="max",
        verbose=True
    )

    # 6. Trainer：根据 epoch_based_training 决定 limit_train_batches
    if args.epoch_based_training:
        limit_batches = 1.0   # 跑完整个 CombinedLoader（mode="min_size"）
    else:
        limit_batches = args.n_iter_per_epoch  # 每轮只跑固定迭代数

    trainer = Trainer(
        max_epochs=args.n_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision='16-mixed',
        deterministic=False,
        callbacks=[checkpoint_callback],
        logger=[wandb_logger, tb_logger],
        limit_train_batches=limit_batches
    )

    # 7. Fit
    trainer.fit(model, datamodule=data_module)

    # 8. Test：在目标域（若指定）或源域验证集上测试
    print("训练完成，开始测试 …")
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        best_model = MInterface.load_from_checkpoint(best_ckpt)
        trainer.test(best_model, datamodule=data_module)
    else:
        print("未找到最佳 checkpoint，直接用当前模型做测试 …")
        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium') 
    parser = configargparse.ArgumentParser(
        description="TransferLearning for OfficeHome Dataset",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    parser.add("--config", is_config_file=True, default='DAN/DAN.yaml', help="YAML 配置文件路径")

    # —— 通用参数 —— #
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader num_workers")
    parser.add_argument("--devices", type=int, default=1, help="使用的 GPU 数量；>1 时启用 DDP")

    # —— 网络 & 对齐损失 —— #
    parser.add_argument("--backbone", type=str, default="resnet50", help="骨干网络名称")
    parser.add_argument("--use_bottleneck", type=bool, default=True, help="是否使用 Bottleneck 层")
    parser.add_argument("--bottleneck_width", type=int, default=256, help="Bottleneck 层的输出维度")
    parser.add_argument("--transfer_loss", type=str, default="mmd",
                        choices=["mmd", "lmmd", "coral", "adv", "daan", "bnm"],
                        help="选择哪种领域对齐损失")

    # —— 数据 & 域设置 —— #
    parser.add_argument("--data_path", type=str, default="/root/datasets/OfficeHome",
                        help="OfficeHome 数据集根目录，包含四个子文件夹")
    parser.add_argument("--src_domain", type=str, default="Art",
                        choices=["Art", "Clipart", "Product", "RealWorld"],
                        help="源域子路径名称")
    parser.add_argument("--tgt_domain", type=str, default="Clipart",
                        choices=["Art", "Clipart", "Product", "RealWorld", "None"],
                        help="目标域子路径名称；若设为 'None' 则只在源域做验证/测试")
    parser.add_argument("--class_num", type=int, default=65,
                        help="分类任务的类别数目；OfficeHome 是 65 类")

    # —— 训练超参 —— #
    parser.add_argument("--batch_size", type=int, default=64, help="训练批大小")
    parser.add_argument("--image_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--n_epoch", type=int, default=30, help="训练最大轮数")
    parser.add_argument("--early_stop", type=int, default=0,
                        help="如果 val_acc 连续若干 epoch 不增则早停；<=0 表示不启用早停")
    parser.add_argument("--epoch_based_training", type=bool, default=False,
                        help="True: 一个 epoch 以最短 loader 耗尽为准；False: 每轮只固定 n_iter_per_epoch 次迭代")
    parser.add_argument("--n_iter_per_epoch", type=int, default=500,
                        help="当 epoch_based_training=False 时，一个 epoch 的迭代次数")

    # —— 优化器 & 学习率调度 —— #
    parser.add_argument("--lr", type=float, default=3e-3, help="基础学习率")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=bool, default=True, help="是否启用 LR Scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.0003, help="LR 衰减中的 gamma 参数")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="LR 衰减中的 decay 指数")

    # —— 对齐损失权重 —— #
    parser.add_argument("--transfer_loss_weight", type=float, default=1.0,
                        help="分类 loss 与 transfer loss 之间的加权比例")

    # —— 增强 & 日志 & 保存路径 —— #
    parser.add_argument("--aug_type", type=str, default="default",
                        choices=["default", "light", "strong"],
                        help="训练阶段的增强策略")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard 日志根目录")
    parser.add_argument("--project_name", type=str, default="OfficeHome_Transfer", help="Project 名称")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="模型 checkpoint 保存目录")

    args = parser.parse_args()

    # 把字符串 "None" 转成 None
    if args.tgt_domain == "None":
        args.tgt_domain = None

    # 设定随机种子
    seed_everything(args.seed, workers=True)

    # 运行训练
    train(args)
