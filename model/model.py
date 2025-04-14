import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, **kwargs):
        image_size = kwargs["image_size"]
        input_size = image_size * image_size * 3
        output_size = kwargs["class_num"]

        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x