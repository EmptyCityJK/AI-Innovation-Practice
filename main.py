import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import DInterface
from model import MInterface
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import wandb

def train(args):
    wandb.init(project="RealWorldClassification", config=args)
    wandb_logger = WandbLogger(project="RealWorldClassification", config=args)
    data_module = DInterface(**args)

    model = MInterface(**args)

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1, verbose=True)

    wandb.init(project='RealWorldClassification', config=args)
    wandb_logger = WandbLogger(project='RealWorldClassification', config=args)
    trainer = Trainer(max_epochs=args['epochs'], callbacks=[checkpoint_callback],
                      logger=wandb_logger)
    trainer.fit(model, data_module)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/", help="Path to image dataset")
    parser.add_argument("--input_size", type=int, default=128, help="Size of the input image to the model")
    parser.add_argument("--image_channels", type=int, default=3, help="Number of channels in the input image")
    parser.add_argument("--output_size", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loaders")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="DCGAN", help="Name of the model to train")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script in: train or predict")
    # parser.add_argument("--scheckpoint_path", type=str, default="", help="Path to the model checkpoint for predictions")

    args = parser.parse_args()
    if args.mode == "train":
        train(vars(args))
    else:
        pass