import matplotlib.pyplot as plt
import numpy as np
from datasets.data_interface import DInterface

def test_augmentations():
    # 初始化数据模块
    datamodule = DInterface(
        data_path="./data",  # 替换为实际路径
        batch_size=32,
        image_size=224,
        num_workers=4,
        aug_type="strong"  # 测试强力增强
    )
    
    # 准备数据集
    datamodule.setup()
    
    # 可视化增强效果
    visualize_augmentations(datamodule.train_dataset)

def visualize_augmentations(dataset, n_samples=3, n_augmentations=5):
    """可视化增强效果"""
    fig, axes = plt.subplots(n_samples, n_augmentations, figsize=(15, 8))
    
    for i in range(n_samples):
        original_img, _ = dataset.dataset[0]  # 获取原始图像
        
        for j in range(n_augmentations):
            # 获取增强后的图像
            img, _ = dataset[0]  # 每次会重新应用随机增强
            
            # 反归一化显示
            img = img.numpy().transpose(1, 2, 0)
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # ImageNet反归一化
            img = np.clip(img, 0, 1)
            
            axes[i,j].imshow(img)
            axes[i,j].axis('off')
            if i == 0:
                axes[i,j].set_title(f'Aug {j+1}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_augmentations()