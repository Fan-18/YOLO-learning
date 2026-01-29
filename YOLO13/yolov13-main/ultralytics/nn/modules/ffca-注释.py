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
    





class Concat2(nn.Module):
    """
    逻辑一致性：对应 FFCA-YOLO 论文中的 CRC (Channel Reweight Concat) 模块。
    实现 2 路输入特征的通道级可学习重权拼接，增强特征融合的表达能力。
    """
    def __init__(self, ch_list, dimension=1):
        # ch_list: 由 YOLO11 的 parse_model 自动传入的输入通道数列表，例如 [128, 256]
        super().__init__()
        self.d = dimension      # 拼接维度，通常为 1 (通道轴)
        self.Channel_all = sum(ch_list)  # 计算所有输入特征的总通道数
        
        # 初始化可学习权重参数 w，初始值全部设为 1.0 (float32 精度)
        # nn.Parameter 确保该参数会被注册到模型中并参与反向传播优化
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4     # 极小值，用于防止归一化计算时除以零

    def forward(self, x):
        # x: 输入 Tensor 列表 [x1, x2]
        # 1. 动态获取每一路当前真实的通道数 (支持模型剪枝后的自动适配)
        c1_real = x[0].shape[1]
        c2_real = x[1].shape[1]
        
        # 2. 动态截取权重 w 并执行论文公式(9)的快速归一化 (Fast Normalized Fusion)
        w = self.w[:(c1_real + c2_real)] 
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # 3. 执行通道重加权 (Reweight)
        # 使用 .view(1, -1, 1, 1) 将 1D 权重向量升维为 4D，触发 PyTorch 广播机制
        # 这种方式比原版的 view(N, H, W, C) 内存重排效率更高，且逻辑等价
        x1 = x[0] * weight[:c1_real].view(1, -1, 1, 1)  # 对第 1 路特征加权
        x2 = x[1] * weight[c1_real:].view(1, -1, 1, 1)  # 对第 2 路特征加权
        
        # 4. 在通道维度上进行物理拼接
        return torch.cat([x1, x2], self.d)


class Concat3(nn.Module):
    """
    逻辑一致性：对应 FFCA-YOLO 论文 CRC 模块，实现 3 路特征的通道重权拼接。
    常用于融合 Backbone、Top-down 路径以及 Bottom-up 路径的同尺度特征。
    """
    def __init__(self, ch_list, dimension=1):
        # ch_list: 3 个输入层的通道列表，例如 [64, 128, 256]
        super().__init__()
        self.d = dimension
        self.Channel_all = sum(ch_list)
        # 初始化 3 路特征所需的总权重空间
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        # x: 包含 3 个 Tensor 的列表 [x1, x2, x3]
        # 1. 动态读取输入尺寸，解决剪枝后通道数变动导致的维度冲突
        c1 = x[0].shape[1]
        c2 = x[1].shape[1]
        c3 = x[2].shape[1]
        
        # 2. 计算归一化后的相对权重比例
        w_active = self.w[:(c1 + c2 + c3)] 
        weight = w_active / (torch.sum(w_active, dim=0) + self.epsilon)
        
        # 3. 定义切分索引，确保权重与对应的输入特征精确匹配
        c1_end = c1
        c2_end = c1 + c2

        # 4. 广播加权运算：(N, C, H, W) * (1, C, 1, 1)
        # 权重 weight 被逻辑上“拉伸”到与 x 的 H, W 尺寸一致，实现逐通道缩放
        x1 = x[0] * weight[:c1_end].view(1, -1, 1, 1)         # 处理第 1 路 (如浅层特征)
        x2 = x[1] * weight[c1_end:c2_end].view(1, -1, 1, 1)  # 处理第 2 路 (如中间融合层)
        x3 = x[2] * weight[c2_end:].view(1, -1, 1, 1)        # 处理第 3 路 (如深层特征)
        
        # 5. 按照指定的维度 (dimension=1) 拼接三路特征
        return torch.cat([x1, x2, x3], self.d)