import os
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split



def data_load(root_path, dir, batch_size, numworker):
    base_transform = transforms.Compose([
        transforms.Resize(256),  # 仅调整尺寸用于后续裁剪
    ])

    dataset = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=base_transform)

    # --- 步骤3：按类别划分索引 ---
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # 对每个类别按比例分割
    train_indices, val_indices, test_indices = [], [], []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        n = len(indices)
        train_end = int(0.6 * n)
        val_end = train_end + int(0.2 * n)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    # --- 步骤4：创建Subset对象 ---
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # --- 步骤5：定义不同子集的Transform ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 步骤6：包装Subset以应用不同Transform ---
    train_dataset = ApplyTransform(train_subset, train_transform)
    val_dataset = ApplyTransform(val_subset, val_test_transform)
    test_dataset = ApplyTransform(test_subset, val_test_transform)
    # 统计信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # --- 步骤7：创建DataLoader ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=numworker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=numworker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=numworker)

    return train_loader, val_loader, test_loader

class ApplyTransform(Subset):
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)

        # Check if the image is a tensor, and only apply the transform if it's a PIL Image or ndarray
        if isinstance(img, torch.Tensor):
            # If it's already a tensor, no need to apply ToTensor again, just return the image
            if self.transform:
                # Apply the transformation that expects a tensor (e.g., Normalize)
                img = self.transform(img)
        else:
            # Apply transform if the image is not a tensor
            if self.transform:
                img = self.transform(img)

        return img, label
