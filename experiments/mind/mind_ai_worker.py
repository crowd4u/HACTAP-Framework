import torch
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary


class MindAIWorker(nn.Module):
    def __init__(self):
        super(MindAIWorker, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(48256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = F.softmax(x, dim=-1)
        return output


def main():
    model = MindAIWorker()
    model.cuda()
    print(model)
    print(summary(model, (3, 122, 110)))


if __name__ == "__main__":
    main()
