import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import DInterface
from model import MInterface
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import wandb

def train(args):
    if args['k_fold'] > 0:
        _cross_validation_train(args)
    else:
        _single_train(args)

# 单独一次训练
def _single_train(args):
    wandb.init(project="RealWorldClassification", config=args)
    wandb_logger = WandbLogger(project="RealWorldClassification", config=args)
    data_module = DInterface(**args)
    model = MInterface(**args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", 
        mode='max', 
        save_top_k=1, 
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=20,
        mode="max",
        verbose=True
    )
    trainer = Trainer(max_epochs=args['epochs'], callbacks=[checkpoint_callback, early_stop_callback],
                      logger=wandb_logger, devices=1, accelerator='gpu')
    
    trainer.fit(model, datamodule=data_module)
    print("训练完成，自动评估测试集...")
    model_path = checkpoint_callback.best_model_path
    if model_path:
        best_model = MInterface.load_from_checkpoint(model_path)
        trainer.test(best_model, datamodule=data_module)
    else:
        print("未保存最佳模型，使用当前模型测试...")
        trainer.test(model, datamodule=data_module)
    wandb.finish()

# 交叉验证
def _cross_validation_train(args):
    fold_results = []
    for fold in range(args['k_fold']):
        args["current_fold"] = fold
        seed_everything(42)
        wandb.init(
            project="RealWorldClassification",
            config=args,
            group=f"{args['k_fold']}-fold-cv",
            name=f"fold-{fold+1}"
        )
        data_module = DInterface(**args)
        model = MInterface(**args)
        # 训练配置
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/fold_{fold+1}",
            monitor="val_acc",
            mode="max",
            filename="best-{epoch}-{val_acc:.2f}"
        )
        wandb_logger = WandbLogger(
            project="RealWorldClassification",
            config=args,
            group=f"CV-{args['k_fold']}fold",
            name=f"fold{fold+1}"
        )
        trainer = Trainer(
            max_epochs=args['epochs'],
            callbacks=[checkpoint_callback, EarlyStopping(monitor="val_acc", patience=20, mode="max")],
            logger=wandb_logger,
            deterministic=True
        )
        
        # 训练验证
        trainer.fit(model, datamodule=data_module)
        model_path = checkpoint_callback.best_model_path
        best_model = MInterface.load_from_checkpoint(model_path)
        trainer.test(best_model, datamodule=data_module)
        fold_results.append(test_result[0]["test_acc"])
        wandb.finish()

    # 输出结果
    final_metrics = {
        "cv_mean_acc": np.mean(fold_results),
        "cv_std_acc": np.std(fold_results),
        "cv_details": fold_results
    }
    print(f"\n{args['k_fold']}-Fold CV Results:\n{final_metrics}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../autodl-tmp/Real World/", help="Path to image dataset")
    parser.add_argument("--image_size", type=int, default=224, help="Size of the input image to the model")
    parser.add_argument("--image_channels", type=int, default=3, help="Number of channels in the input image")
    parser.add_argument("--class_num", type=int, default=65, help="Dimensionality of the latent space")
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the model')
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for data loaders")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--lr_scheduler", type=bool, default=True, help="Use learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument('--backbone', type=str, default='vgg13', help='Backbone model to use')
    parser.add_argument("--model_name", type=str, default="VGG", help="Name of the model to train")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script in: train or predict")
    parser.add_argument("--k_fold", type=int, default=0, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--aug_type", type=str, default="default", help="Type of augmentation to use: default or light or strong")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup/Cutmix的alpha参数（0表示禁用）")
    parser.add_argument("--cutmix_prob", type=float, default=0.5, help="执行Cutmix的概率（剩余概率使用Mixup）")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin value for ArcFace (recommended: 0.3-0.7, typical 0.5)")
    parser.add_argument("--scale", type=int, default=32, help="Scale factor for ArcFace (recommended: 32-128, typical 64)")
    # parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint for predictions")

    args = parser.parse_args()
    if args.mode == "train":
        train(vars(args))
    else:
        pass