import torch
import torch.nn as nn
import torch.nn.functional as F
import model.backbone.ResNet
from math import pi

class ArcFaceLayer(nn.Module):
    """ArcFace 决策边界优化层"""
    def __init__(self, input_dim, num_classes, margin=0.5, scale=64):
        super(ArcFaceLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.W = nn.Parameter(torch.Tensor(input_dim, num_classes))
        nn.init.xavier_normal_(self.W)

    def forward(self, x, labels=None):
        # 特征和权重归一化
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=0)
        cos_theta = torch.mm(x_norm, W_norm)
        
        if labels is None:
            return cos_theta * self.scale

        # 训练模式：计算带 margin 的 logits
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.one_hot(labels, num_classes=self.num_classes)
        cos_theta_m = torch.cos(torch.min(theta + self.margin * one_hot, torch.ones_like(theta) * pi))
        logits = self.scale * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)
        return logits

class DeepDBC(nn.Module):
    """DeepDBC 模块"""
    def __init__(self, input_dim, **kwargs):
        super(DeepDBC, self).__init__()
        self.hidden_dim = kwargs['hidden_dim']
        self.num_classes = kwargs['class_num']
        self.margin = kwargs['margin']
        self.scale = kwargs['scale']

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # ArcFace 层
        self.arcface = ArcFaceLayer(
            self.hidden_dim, 
            self.num_classes, 
            margin=self.margin,
            scale=self.scale  
        )

    def forward(self, x, labels=None):
        features = self.feature_extractor(x)
        return self.arcface(features, labels)

class ResNetDeepDBC(nn.Module):
    """完整模型"""
    def __init__(self, **kwargs):
        super(ResNetDeepDBC, self).__init__()
        # 从 kwargs 获取参数
        self.backbone = getattr(model.resnet, kwargs['backbone'])(pretrained=kwargs.get('pretrained', False))
        self.deepdbc = DeepDBC(input_dim=2048, **kwargs)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        return self.deepdbc(features, labels)