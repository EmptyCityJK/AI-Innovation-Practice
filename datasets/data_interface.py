import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from .transforms import Transforms

class TransformDataset(Dataset):
    """
    将 Subset 或 ImageFolder 包一层，好方便在 __getitem__ 时给 PIL Image 贴上 transform。
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]   # x: PIL Image, y: int
        if self.transform:
            x = self.transform(x)   # 将 PIL Image 转为 Tensor
        return x, y

    def __len__(self):
        return len(self.subset)


class DInterface(pl.LightningDataModule):
    """
    支持领域自适应的 DataModule，同时提供“无限迭代”版本的 source 与 target loader。
    - source_train_dataloader(): 有标签的源域数据，迭代完整后会在 Lightning 中被手动 cycle。
    - target_train_dataloader(): 无标签的目标域数据，同样可以无限 cycle。
    - val_dataloader(), test_dataloader(): 都是目标域带标签的验证/测试数据。
    """

    def __init__(
        self,
        data_path: str,
        source_domain: str = "RealWorld",
        target_domain: str = "Clipart",
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        aug_type: str = "default",
        seed: int = 42,
        epoch_based: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.aug_type = aug_type
        self.seed = seed
        self.epoch_based = epoch_based

        # OfficeHome 固定 65 类
        self.num_classes = 65

        # 根据 aug_type 选择训练阶段 transform
        if self.aug_type == "light":
            self.train_transform = Transforms.get_light_augment_transform(self.image_size)
        elif self.aug_type == "strong":
            self.train_transform = Transforms.get_strong_augment_transform(self.image_size)
        else:
            self.train_transform = Transforms.get_default_train_transform(self.image_size)

        # 验证/测试阶段统一使用标准化 transform
        self.val_test_transform = Transforms.get_val_test_transform(self.image_size)

        # 以下属性在 setup() 中会被赋值
        self.source_train_dataset = None
        self.target_train_dataset = None
        self.target_test_dataset = None

    def prepare_data(self):
        # OfficeHome 数据集一般手动下载，此处无需自动下载
        pass

    def setup(self, stage=None):
        # —— Step 1: 源域全部作为有标签 train —— #
        source_dir = os.path.join(self.data_path, self.source_domain)
        assert os.path.isdir(source_dir), f"源域目录 '{source_dir}' 不存在"
        full_source = datasets.ImageFolder(root=source_dir, transform=None)
        self.source_train_dataset = TransformDataset(full_source, transform=self.train_transform)

        # —— Step 2: 目标域无标签 train，以及带标签 val/test —— #
        if self.target_domain is not None:
            target_dir = os.path.join(self.data_path, self.target_domain)
            assert os.path.isdir(target_dir), f"目标域目录 '{target_dir}' 不存在"
            full_target = datasets.ImageFolder(root=target_dir, transform=None)
            self.target_train_dataset = TransformDataset(full_target, transform=self.train_transform)
            self.target_test_dataset  = TransformDataset(full_target, transform=self.val_test_transform)
        else:
            self.target_train_dataset = None
            self.target_test_dataset  = None

    def source_train_dataloader(self):
        """
        返回源域（有标签）训练集 DataLoader。后续由 LightningModule 自行 “cycle” 以实现无限加载。
        """
        return DataLoader(
            self.source_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )

    def target_train_dataloader(self):
        """
        返回目标域（无标签）训练集 DataLoader。同样会被无限 “cycle”。
        """
        if self.target_train_dataset is None:
            return None

        return DataLoader(
            self.target_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )

    def train_dataloader(self):
        """
        Lightning 依然需要 train_dataloader() 接口，返回源域 DataLoader 即可。
        在 LightningModule 中，我们会忽略传入的 batch，改为手动从 source_iter/target_iter 拿数据。
        """
        return self.source_train_dataloader()

    def val_dataloader(self):
        """
        验证时，只在目标域带标签的 val/test 数据上评估（此处直接复用 test）。
        """
        return DataLoader(
            self.target_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        """
        测试时，同样在目标域带标签数据上评估。
        """
        return DataLoader(
            self.target_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
