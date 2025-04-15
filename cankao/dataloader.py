import os
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

def data_load(root_path, dir, batch_size, numworker):
    # 定义训练、验证、测试的变换
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集，直接应用不同的 transform
    train_dataset = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=val_test_transform)

    # 按类别划分索引
    class_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # 对每个类别按比例分割
    train_indices, val_indices, test_indices = [], [], []
    for label, indices in class_indices.items():
        np.random.seed(42)  # 固定随机种子以确保可重复性
        np.random.shuffle(indices)
        n = len(indices)
        train_end = int(0.6 * n)
        val_end = train_end + int(0.2 * n)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    # 创建 Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 统计信息
    print(f"训练集样本数: {len(train_subset)}")
    print(f"验证集样本数: {len(val_subset)}")
    print(f"测试集样本数: {len(test_subset)}")

    # 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=numworker)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=numworker)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=numworker)

    return train_loader, val_loader, test_loader