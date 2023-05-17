import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import data_utils


class Net(nn.Module):
    def __init__(self, num_channels, num_classes, model_type, location_len, device, use_location):
        super(Net, self).__init__()
        self.location_len = location_len
        self.device = device
        self.use_location = use_location
        if use_location:
            num_resnet_fc = num_classes
        else:
            num_resnet_fc = 512
        if model_type == 'resnet18':
            self.resnet = models.resnet18(num_classes=num_resnet_fc)
        elif model_type == 'resnet34':
            self.resnet = models.resnet34(num_classes=num_resnet_fc)
        elif model_type == 'resnet50':
            self.resnet = models.resnet50(num_classes=num_resnet_fc)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fc_a = nn.Sequential(
            nn.Linear(num_resnet_fc + self.location_len, 2048),
            nn.ReLU()
        )
        self.fc_b = nn.Linear(2048, num_classes)

    def forward(self, x, location):
        x = self.resnet(x)
        if not self.use_location:
            return x
        location_code = data_utils.get_location_code(location[:, 0], location[:, 1], self.location_len)
        location_code = location_code.to(self.device)
        x = torch.concat([x, location_code], dim=1)
        x = self.fc_a(x)
        x = self.fc_b(x)
        return x


# 测试
if __name__ == '__main__':
    pass
