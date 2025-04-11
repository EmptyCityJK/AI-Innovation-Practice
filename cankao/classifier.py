import torch
import torch.nn as nn
import model

class Model4Classifier(nn.Module):
    # Resnet+MLP
    def __init__(self, args):
        super(Model4Classifier, self).__init__()
        self.class_num = args.class_num
        self.hidden_dim = args.hidden_dim

        # classifier
        self.backbone = getattr(model, args.backbone)(pretrained=False)
        self.classifier = nn.Sequential()
        # 全连接层1
        self.classifier.add_module('fc1',nn.Linear(2048, self.hidden_dim))
        self.classifier.add_module('batchnorm1d', nn.BatchNorm1d(self.hidden_dim))
        self.classifier.add_module('relu',nn.ReLU(inplace=True))
        self.classifier.add_module('dropout',nn.Dropout(p=0.5))
        self.classifier.add_module('fc2',nn.Linear(self.hidden_dim,self.class_num))

    def forward(self, x):
        feature_map = self.backbone(x)
        output = self.classifier(feature_map)

        return output





