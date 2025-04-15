import os
import torch
from torch import nn
import argparse
import dataloader
from classifier import Model4Classifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Course_for_OfficeHome')
# 模型的基本参数
parser.add_argument('--backbone', type=str, default='resnet50')#resnext50_32x4d resnet50
parser.add_argument('--hidden_dim', type=int, default=512) #分类头全连接层的神经元数量
# 数据的基本参数
parser.add_argument('--data_path', type=str, default="../../autodl-tmp/")
parser.add_argument('--fold_name', type=str, default="Real World")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--class_num', type=int, default=65)
parser.add_argument('--workers', type=int, default=16)
# 训练阶段的基本参数
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--iters_per_epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=1e-4)
parser.add_argument('--lr_scheduler', type=bool, default=True)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=1e-3)
parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    ######## 数据
    root_dir = args.data_path
    fold_name = args.fold_name
    train_loader, val_loader, test_loader = dataloader.data_load(root_dir,fold_name,args.batch_size, args.workers)

    ##### 模型
    model = Model4Classifier(args).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        [{'params':model.backbone.parameters(), 'lr':args.lr},
         {'params':model.classifier.parameters(), 'lr':args.lr}
        ],  lr=args.lr, weight_decay=args.lr_decay)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs * 0.5), eta_min=1e-5)

    ##### 训练参数初始化
    best_val_acc = 0.0
    train_loss_history, val_acc_history = [], []

    for epoch in range(args.epochs):
        #### 模型训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            ### 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            ### 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### 统计训练指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 计算epoch训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_loss_history.append(train_loss)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        #### 模型验证阶段
        model.eval()  # Switch to evaluation mode
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():  # 取消梯度计算
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                val_outputs = model(inputs)
                loss = criterion(val_outputs, labels)
                val_loss += loss.item()

                _, val_predicted = torch.max(val_outputs, 1)
                correct += (val_predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_acc_history.append(val_acc)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')  # Save the model

        # Update learning rate if scheduler is enabled
        if args.lr_scheduler:
            scheduler.step()

    # Print training and validation histories
    print("Training History:")
    print(f"Train Loss: {train_loss_history}")
    print(f"Validation Accuracy: {val_acc_history}")

    # Final testing
    model.load_state_dict(torch.load('best_model.pth'))  # Load the best model
    model.eval()  # Switch to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)  # Get model outputs
            loss = criterion(outputs, labels)  # Compute loss
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get predicted classes
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total number of samples

    test_acc = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")