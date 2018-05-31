'''GoogLeNet with PyTorch'''
import torch
import torch.nn as nn

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
        super(PoseNet, self).__init__()

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
        self.fc = nn.Linear(1024, 2048)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.cls_fc_pose_xyz = nn.Linear(2048, 3)
        self.cls_fc_pose_wpqr = nn.Linear(2048, 4)

    def forward(self, x):
        out = self.pre_layers(x)

        out = self.a3(out)
        out = self.b3(out)

        out = self.max_pool(out)

        out = self.a4(out)
        cls1_pool = self.avg_pool(out)
        cls1_fc1 = self.fc(cls1_pool)
        cls1_fc1 = self.relu(cls1_fc1)
        cls1_fc1 = self.dropout(cls1_fc1)
        cls1_fc_pose_xyz = self.cls_fc_pose_xyz(cls1_fc1)
        cls1_pose_wpqr = self.cls_fc_pose_wpqr(cls1_fc1)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        cls2_pool = self.avg_pool(out)
        cls2_fc1 = self.fc(cls2_pool)
        cls2_fc1 = self.relu(cls2_fc1)
        cls2_fc1 = self.dropout(cls2_fc1)
        cls2_fc_pose_xyz = self.cls_fc_pose_xyz(cls2_fc1)
        cls2_pose_wpqr = self.cls_fc_pose_wpqr(cls2_fc1)
        out = self.e4(out)

        out = self.max_pool(out)

        out = self.a5(out)
        out = self.b5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        cls3_pool = self.avg_pool(out)
        cls3_fc1 = self.fc(cls3_pool)
        cls3_fc1 = self.relu(cls3_fc1)
        cls3_fc1 = self.dropout(cls3_fc1)
        cls3_fc_pose_xyz = self.cls_fc_pose_xyz(cls3_fc1)
        cls3_pose_wpqr = self.cls_fc_pose_wpqr(cls3_fc1)

        out = [
            [cls1_fc_pose_xyz, cls1_pose_wpqr],
            [cls2_fc_pose_xyz, cls2_pose_wpqr],
            [cls3_fc_pose_xyz, cls3_pose_wpqr]
        ]
        return out


net = GoogLeNet().cuda()
x = torch.randn(1, 3, 32, 32)
print(x)
print(x.size())
y = net(Variable(x).cuda())

print(y.size())
