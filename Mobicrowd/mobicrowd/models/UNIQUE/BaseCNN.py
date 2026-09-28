import torch.nn as nn # type: ignore
from torchvision import models # type: ignore
from torchvision.models import ResNet34_Weights,ResNet18_Weights # type: ignore
from BCNN import BCNN
from torch import nn
import torch


class BaseCNN(nn.Module):
    representation_type: str
    backbone_type: str
    fc_enabled: bool
    std_modeling: bool

    def __init__(self, backbone: str = 'resnet18', representation: str = 'BCNN', fc: bool = True, std_modeling: bool = False):
        super().__init__()

        self.backbone_type = backbone
        self.representation_type = representation
        self.fc_enabled = fc
        self.std_modeling = std_modeling

        if self.backbone_type == 'resnet18':
            self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif self.backbone_type == 'resnet34':
            self.backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        if self.representation_type == 'BCNN':
            self.representation = BCNN()
            self.fc = nn.Linear(512 * 512, 2 if self.std_modeling else 1)
        else:
            self.fc = nn.Linear(512, 2 if self.std_modeling else 1)

        if self.fc_enabled:
            for param in self.backbone.parameters():
                param.requires_grad = False
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if self.representation_type == 'BCNN':
            x = self.representation(x)
        else:
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        if self.std_modeling:
            mean = x[:, 0]
            t = x[:, 1]
            var = nn.functional.softplus(t)
        else:
            mean = x[:, 0]
            var = torch.tensor(0.0, device=mean.device).expand_as(mean)  # dummy tensor to unify type

        return mean, var
