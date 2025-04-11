import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, **kwargs):
        input_size = kwargs["input_size"]
        output_size = kwargs["output_size"]

        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x