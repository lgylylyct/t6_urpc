import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#summary(model, input_size=(channels, H, W))

if __name__ == '__main__':
    resnet18 = models.resnet18().cuda() # 不加.cuda()会报错
    summary(resnet18, input_size=(2, 3, 64, 64))

