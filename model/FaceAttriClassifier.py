import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import sys, os

sys.path.append(os.getcwd())

from config import read_yaml
from torchvision.models import mobilenetv3
from model.Shufflenetv2 import ShuffleNetV2
import pdb


def get_output_channel(model):
    input = torch.randn(1, 3, 224, 224)

    output = model(input)
    output_channels = []
    for out in output:
        output_channels.append(out.shape[1])

    return output_channels


class BaseModel(nn.Module):
    def __init__(self, backbone):
        super(BaseModel, self).__init__()
        # if backbone == 'mobilenet':
        #     self.encoder = smp.encoders.get_encoder(
        #         name = 'timm-mobilenetv3_large_100',
        #         weights='imagenet'
        #     )
        if backbone == "shufflenetv2":
            self.encoder = ShuffleNetV2()

        # 获取encoder输出维度
        self.output_channel = get_output_channel(self.encoder)

    def forward(self, x):
        output = self.encoder(x)
        return output


class Reg_Head(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Reg_Head, self).__init__()
        # head头的编写
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            # nn.Linear(input_channel, 256), # 960,256
            # nn.BatchNorm1d(256, eps=0.1),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=0.1),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, output_channel),
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=100,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=100, out_channels=10, kernel_size=1, stride=1, padding=0
            ),
            nn.Conv2d(
                in_channels=10, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(self, x):
        x = self.pool(x)
        output = self.head(x)
        output = torch.flatten(output, 1)
        return output


class Class_Head(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Class_Head, self).__init__()
        # head头的编写
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(input_channel, 128),  # 960,256
            nn.BatchNorm1d(128, eps=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_channel),
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        output = self.head(x)
        output = torch.flatten(output, 1)
        return output


class FaceAttriClassifier(nn.Module):
    def __init__(self, cfg):
        super(FaceAttriClassifier, self).__init__()
        self.cfg = cfg
        self.backbone = BaseModel(backbone=cfg.model.backbone)
        self.input_channel = self.backbone.output_channel[-1]  # 取最后一个output
        self.output_channel = cfg.model.num_class
        self.head = Class_Head(self.input_channel, self.output_channel)

    def forward(self, x):
        encoder_feat = self.backbone(x)
        output = self.head(encoder_feat[-1])

        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = read_yaml()
    model = FaceAttriClassifier(cfg).to(device)
    model.eval()
    input = torch.randn(1, 3, 112, 112).to(device)
    output = model(input)
    summary(model, (3, 112, 112))
