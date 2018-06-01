'''GoogLeNet with PyTorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class InceptionV1(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionV1, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __int__(self):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.LocalResponseNorm(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.a3 = InceptionV1(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionV1(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = InceptionV1(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionV1(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionV1(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionV1(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionV1(528, 256, 160, 320, 32, 128, 128)

        self.a5 = InceptionV1(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionV1(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avg_pool5x5 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.fc = nn.Linear(1024, 1000)
        self.fc_128 = nn.Linear(128, 1024)
        self.soft_max = nn.Softmax2d()
        self.conv1x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv2d(528, 128, kernel_size=1, stride=1)

        self.dropout_4 = nn.Dropout(p=0.4)
        self.dropout_7 = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pre_layers(x)

        out = self.a3(out)
        out = self.b3(out)

        out = self.max_pool(out)

        out = self.a4(out)

        # 5x5 pooling
        cls1_pool = self.avg_pool5x5(out)
        # 1x1 conv
        cls1_reduction = self.conv1x1(cls1_pool)
        cls1_reduction = self.relu(cls1_reduction)
        cls1_reduction = cls1_reduction.view(cls1_reduction.size(0), -1)
        # 128 fc
        cls1_fc1 = self.fc_128(cls1_reduction)
        cls1_fc1 = self.relu(cls1_fc1)
        # 0.7 dropout
        cls1_fc1 = self.dropout_7(cls1_fc1)
        # 1024 fc
        cls1_fc2 = self.fc(cls1_fc1)
        # softmax
        cls1 = self.soft_max(cls1_fc2)

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)

        # 5x5 pooling
        cls2_pool = self.avg_pool5x5(out)
        # 1x1 conv
        cls2_reduction = self.conv1x1_2(cls2_pool)
        cls2_reduction = self.relu(cls2_reduction)
        cls2_reduction = cls2_reduction.view(cls2_reduction.size(0), -1)
        # 128 fc
        cls2_fc1 = self.fc_128(cls2_reduction)
        cls2_fc1 = self.relu(cls2_fc1)
        # 0.7 dropout
        cls2_fc1 = self.dropout_7(cls2_fc1)
        # 1024 fc
        cls2_fc2 = self.fc(cls2_fc1)
        # softmax
        cls2 = self.soft_max(cls2_fc2)

        out = self.e4(out)

        out = self.max_pool(out)

        out = self.a5(out)
        out = self.b5(out)
        cls3_pool = self.avg_pool(out)
        cls3_fc = self.fc(cls3_pool)
        cls3_fc = self.relu(cls3_fc)
        cls3_fc = self.dropout_4(cls3_fc)
        cls3 = self.soft_max(cls3_fc)

        return cls1, cls2, cls3
