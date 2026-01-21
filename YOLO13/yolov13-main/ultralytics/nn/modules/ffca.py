import torch
import torch.nn as nn
from .conv import Conv  # 确保导入了 Ultralytics 的标准卷积类

class FEM(nn.Module):
    def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        inter_planes = c1 // map_reduce

        # 定义统一的激活函数，确保与原版 ReLU 一致
        raw_relu = nn.ReLU(inplace=True)
        
        # 分支0：标准 3x3 卷积 [cite: 218]
        self.branch0 = nn.Sequential(
            Conv(c1, 2 * inter_planes, 1, stride, act=raw_relu),
            Conv(2 * inter_planes, 2 * inter_planes, 3, 1, act=False)
        )
        
        # 分支1：非对称卷积 + 空洞卷积 (Dilation=5) [cite: 219]
        self.branch1 = nn.Sequential(
            Conv(c1, inter_planes, 1, 1),
            Conv(inter_planes, (inter_planes // 2) * 3, (1, 3), stride, p=(0, 1), act=raw_relu),
            Conv((inter_planes // 2) * 3, 2 * inter_planes, (3, 1), 1, p=(1, 0), act=raw_relu),
            nn.Conv2d(2 * inter_planes, 2 * inter_planes, 3, 1, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(2 * inter_planes)
        )
        
        # 分支2：非对称卷积 + 空洞卷积 (Dilation=5) [cite: 219]
        self.branch2 = nn.Sequential(
            Conv(c1, inter_planes, 1, 1),
            Conv(inter_planes, (inter_planes // 2) * 3, (3, 1), stride, p=(1, 0), act=raw_relu),
            Conv((inter_planes // 2) * 3, 2 * inter_planes, (1, 3), 1, p=(0, 1), act=raw_relu),
            nn.Conv2d(2 * inter_planes, 2 * inter_planes, 3, 1, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(2 * inter_planes)
        )

        self.ConvLinear = Conv(6 * inter_planes, c2, 1, 1, act=False)
        self.shortcut = Conv(c1, c2, 1, stride, act=False)
        self.relu = nn.ReLU(inplace=True) # FFCA 原文使用 ReLU 

    def forward(self, x):
        # 对应公式 (1) 的 Concatenation 和 Addition [cite: 220, 221]
        out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        out = self.ConvLinear(out)
        return self.relu(out * self.scale + self.shortcut(x))