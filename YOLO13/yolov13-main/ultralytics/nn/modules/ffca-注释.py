import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class FEM(nn.Module):
    """
    Feature Enhancement Module (FEM) - 完美对应 FFCA 原作者逻辑版
    
    参数含义解析：
    c1 (int): 输入通道数。代表上一层传过来的特征图厚度。
    c2 (int): 输出通道数。代表本模块处理完后，传给下一层的特征图厚度。
    stride (int): 步长。默认为1。如果为2，则会进行下采样，特征图尺寸减半。
    scale (float): 残差缩放因子。用于控制增强特征对原始特征的修改程度。
    map_reduce (int): 通道压缩率。用于计算内部中间通道，降低计算量。
    """
    def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale  # 对应公式中的系数，平衡原始信息与增强信息
        
        # inter_planes: 模块内部使用的基准通道数，通过 map_reduce 压缩输入通道
        # 这样做是为了在并行多个分支时，总计算量（GFLOPs）不至于爆炸。
        inter_planes = c1 // map_reduce 
        
        # 统一使用 ReLU 激活函数，确保与 FFCA 原论文设计的非线性变换一致
        raw_relu = nn.ReLU(inplace=True)

        # 分支0 (Branch 0)：标准局部特征提取
        self.branch0 = nn.Sequential(
            # 1x1 卷积：将通道调整为 2 * inter_planes，并进行初步跨通道信息融合
            Conv(c1, 2 * inter_planes, 1, stride, act=raw_relu),
            # 3x3 卷积：捕捉局部空间特征。act=False 保持线性输出，等待汇总。
            Conv(2 * inter_planes, 2 * inter_planes, 3, 1, act=False) 
        )
        
        # 分支1 (Branch 1)：横向优先的非对称卷积 + 大感受野空洞卷积
        self.branch1 = nn.Sequential(
            Conv(c1, inter_planes, 1, 1, act=raw_relu),
            # (1, 3) 卷积：专注于捕捉横向纹理特征（如马路、车辆轮廓）
            Conv(inter_planes, (inter_planes // 2) * 3, (1, 3), stride, p=(0, 1), act=raw_relu),
            # (3, 1) 卷积：补充纵向信息，构建非对称感受野
            Conv((inter_planes // 2) * 3, 2 * inter_planes, (3, 1), 1, p=(1, 0), act=raw_relu),
            # 空洞卷积 (Dilation=5)：在不增加参数的前提下，将有效感受野扩大到 11x11。
            # padding=5 是为了抵消空洞带来的尺寸缩小，保证输出尺寸与 shortcut 对齐。
            Conv(2 * inter_planes, 2 * inter_planes, 3, 1, p=5, d=5, act=False)
        )
        
        # 分支2 (Branch 2)：纵向优先的非对称卷积 + 大感受野空洞卷积
        self.branch2 = nn.Sequential(
            Conv(c1, inter_planes, 1, 1, act=raw_relu),
            # (3, 1) 卷积：优先捕捉纵向纹理特征（如电线杆、建筑物边缘）
            Conv(inter_planes, (inter_planes // 2) * 3, (3, 1), stride, p=(1, 0), act=raw_relu),
            # (1, 3) 卷积：互补横向特征
            Conv((inter_planes // 2) * 3, 2 * inter_planes, (1, 3), 1, p=(0, 1), act=raw_relu),
            # 同上，利用空洞卷积获取全局上下文，帮助识别 VisDrone 数据集中的复杂背景
            Conv(2 * inter_planes, 2 * inter_planes, 3, 1, p=5, d=5, act=False)
        )

        # ConvLinear：通道汇总层。将 3 个分支拼接后的 6 倍厚度特征压缩回目标 c2 通道
        self.ConvLinear = Conv(6 * inter_planes, c2, 1, 1, act=False)
        
        # shortcut：残差路径。如果输入输出通道不一致或有步长，通过 1x1 卷积进行对齐
        self.shortcut = Conv(c1, c2, 1, stride, act=False)
        
        # 最后的 ReLU：对应公式中的 Addition 后接 Activation，实现残差学习的闭环
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        """
        前向传播逻辑：
        1. cat: 将局部、横向、纵向以及大感受野的特征在通道维度上并列排放。
        2. ConvLinear: 学习三个分支特征的权重并进行融合降维。
        3. Addition: 将学习到的“残差”（增强特征）以 10% 的比例叠加到“基底”（原始特征）上。
        """
        out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        out = self.ConvLinear(out)
        return self.relu(out * self.scale + self.shortcut(x))