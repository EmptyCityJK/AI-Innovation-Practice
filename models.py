import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones


class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss

        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        # 1) backbone 提取特征:
        source = self.base_network(source)  # (batch_s, feat_dim)
        target = self.base_network(target)  # (batch_t, feat_dim)

        if self.use_bottleneck:
            source = self.bottleneck_layer(source)  # (batch_s, bottleneck_width)
            target = self.bottleneck_layer(target)  # (batch_t, bottleneck_width)

        # 2) 分类 head：
        source_clf = self.classifier_layer(source)  # (batch_s, num_class)
        clf_loss = self.criterion(source_clf, source_label)

        # 3) 对齐（transfer）loss
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        """
        返回一个参数组列表，为不同子网络分配不同的 lr：
        - backbone: 0.1 * initial_lr
        - classifier_layer: 1.0 * initial_lr
        - bottleneck_layer (若有): 1.0 * initial_lr
        - 对于 'adv' 或 'daan'，再加上 domain classifier 的参数
        """
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        # 预测函数，仅用于 test 阶段
        features = self.base_network(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        # 对于 DAAN, 它内部需要随着 epoch 进度更新 dynamic factor
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
