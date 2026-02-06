import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.nn.modules.conv import autopad, Conv 


# 定义 SimAM 无参注意力模块 (即插即用，不增加参数量)
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class FEM_v2(nn.Module):
    def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8):
        super(FEM_v2, self).__init__()
        self.scale = scale
        inter_planes = c1 // map_reduce

        # 1. 使用 SiLU 替换 ReLU，与 YOLO11 保持一致
        self.act = nn.SiLU() 
        
        # 分支0：基础特征 (保持不变)
        self.branch0 = nn.Sequential(
            Conv(c1, 2 * inter_planes, 1, stride, act=self.act),
            Conv(2 * inter_planes, 2 * inter_planes, 3, 1, act=False)
        )
        
        # 分支1：中等感受野 (修改：Dilation=3)
        self.branch1 = nn.Sequential(
            Conv(c1, inter_planes, 1, 1, act=self.act),
            Conv(inter_planes, (inter_planes // 2) * 3, (1, 3), stride, p=(0, 1), act=self.act),
            Conv((inter_planes // 2) * 3, 2 * inter_planes, (3, 1), 1, p=(1, 0), act=self.act),
            # 改进：Dilation 从 5 改为 3，填充改为 3
            nn.Conv2d(2 * inter_planes, 2 * inter_planes, 3, 1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(2 * inter_planes)
        )
        
        # 分支2：大感受野 (保持 Dilation=5)
        self.branch2 = nn.Sequential(
            Conv(c1, inter_planes, 1, 1, act=self.act),
            Conv(inter_planes, (inter_planes // 2) * 3, (3, 1), stride, p=(1, 0), act=self.act),
            Conv((inter_planes // 2) * 3, 2 * inter_planes, (1, 3), 1, p=(0, 1), act=self.act),
            nn.Conv2d(2 * inter_planes, 2 * inter_planes, 3, 1, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(2 * inter_planes)
        )

        # 2. 融合层
        self.ConvLinear = Conv(6 * inter_planes, c2, 1, 1, act=False)
        self.shortcut = Conv(c1, c2, 1, stride, act=False)
        
        # 3. 引入 SimAM 注意力，增强多尺度特征融合
        self.attention = SimAM()

    def forward(self, x):
        # 拼接三个分支
        cat_out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        
        # 经过线性层压缩
        out = self.ConvLinear(cat_out)
        
        # 注意力加权 (这是新增的，让网络知道哪个 dilation 分支更重要)
        out = self.attention(out)
        
        # 残差连接 (使用 SiLU)
        return self.act(out * self.scale + self.shortcut(x))


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
    



class FFM_Concat2(nn.Module):
    """逻辑一致性：对应论文 CRC 模块，实现 2 路特征的通道重权拼接 (Channel Reweight Concat) """
    def __init__(self, c1, dimension=1):
        # c1 在 YOLO11 中会自动传入输入通道列表，例如 [128, 256]
        super().__init__()
        self.d = dimension
        self.c1 = c1  # 输入通道数列表
        self.Channel_all = sum(c1)
        # 初始化可学习权重为 1
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001 # 

    def forward(self, x):
        # x 为输入 Tensor 列表: [x1, x2]
        # 对应论文公式 (9)：归一化权重计算 
        # 使用 ReLU 确保权重非负（可选，但论文逻辑建议）或直接使用原逻辑
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # 按照各输入层的通道数进行切分并加权
        # YOLO11 默认格式为 (N, C, H, W)，使用 view(1, -1, 1, 1) 进行广播相乘最准确
        x1 = x[0] * weight[:self.c1[0]].view(1, -1, 1, 1)
        x2 = x[1] * weight[self.c1[0]:].view(1, -1, 1, 1)
        
        return torch.cat([x1, x2], self.d)

class FFM_Concat3(nn.Module):
    """逻辑一致性：实现 3 路特征的通道重权拼接 [cite: 258]"""
    def __init__(self, c1, dimension=1):
        # c1 为 3 个输入层的通道列表，例如 [64, 128, 256]
        super().__init__()
        self.d = dimension
        self.c1 = c1
        self.Channel_all = sum(c1)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # x 为输入 Tensor 列表: [x1, x2, x3]
        weight = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        
        c1_end = self.c1[0]
        c2_end = self.c1[0] + self.c1[1]

        x1 = x[0] * weight[:c1_end].view(1, -1, 1, 1)
        x2 = x[1] * weight[c1_end:c2_end].view(1, -1, 1, 1)
        x3 = x[2] * weight[c2_end:].view(1, -1, 1, 1)
        
        return torch.cat([x1, x2, x3], self.d)
    


class Concat2(nn.Module):
    """逻辑一致性：对应论文 CRC 模块，实现 2 路特征的通道重权拼接 (Channel Reweight Concat) """
    def __init__(self, ch_list, dimension=1): # 把 c1 改名为 ch_list 就不纠结了
        super().__init__()
        self.d = dimension
        # ch_list 就是 [C1, C2]
        self.Channel_all = sum(ch_list) 
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # 即使 c1 在初始化时是确定的，但在 forward 里
        # 依然建议动态获取 x[0] 和 x[1] 的通道，以确保【剪枝】后不会报错
        c1_real = x[0].shape[1]
        c2_real = x[1].shape[1]
        
        w = self.w[:(c1_real + c2_real)] # 动态截取，这才是灵魂
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # 使用广播机制进行加权，比原版的 view 转换快得多
        x1 = x[0] * weight[:c1_real].view(1, -1, 1, 1)
        x2 = x[1] * weight[c1_real:].view(1, -1, 1, 1)
        
        return torch.cat([x1, x2], self.d)


    
class Concat3(nn.Module):
    """逻辑一致性：对应 FFCA-YOLO 论文 CRC 模块，实现 3 路特征的通道重权拼接"""
    def __init__(self, ch_list, dimension=1):
        # ch_list 为 3 个输入层的通道列表，由 tasks.py 自动传入，例如 [64, 128, 256]
        super().__init__()
        self.d = dimension
        # 使用 ch_list 计算初始化总通道数
        self.Channel_all = sum(ch_list)
        # 初始化可学习权重 w
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # 1. 动态获取每一路输入的真实通道数（这一步是支持【剪枝】的关键）
        c1 = x[0].shape[1]
        c2 = x[1].shape[1]
        c3 = x[2].shape[1]
        
        # 2. 动态截取权重 w 并进行快速归一化（与 FFM_Concat3 逻辑完全一致）
        w_active = self.w[:(c1 + c2 + c3)] 
        weight = w_active / (torch.sum(w_active, dim=0) + self.epsilon)
        
        # 3. 设置切分点，确保权重精准对应
        c1_end = c1
        c2_end = c1 + c2

        # 4. 执行通道重加权 (使用广播机制 view(1, -1, 1, 1)，效率高于原版 view 重排)
        # 结果与 (weight[:c1] * x[0].view(N, H, W, C)).view(N, C, H, W) 完全等价
        x1 = x[0] * weight[:c1_end].view(1, -1, 1, 1)
        x2 = x[1] * weight[c1_end:c2_end].view(1, -1, 1, 1)
        x3 = x[2] * weight[c2_end:].view(1, -1, 1, 1)
        
        # 5. 最终特征拼接
        return torch.cat([x1, x2, x3], self.d)



class Conv_withoutBN(nn.Module):
    """
    不带 BatchNorm 的标准卷积模块
    适用于：注意力机制的中间层、投影层等不需要 BN 的场景
    """
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=True):
        super().__init__()
        # ⚠️ 修改点 1: 增加了 bias 参数并默认为 True。
        # 在没有 BN 的情况下，卷积层通常需要偏置(bias)来学习截距，否则拟合能力会受限。
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        
        # ⚠️ 修改点 2: 激活函数逻辑优化，兼容 YOLO11 写法
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))   
    

class Conv_withoutBN2(nn.Module):
    """
    不带 BatchNorm 的标准卷积模块
    适用于：注意力机制的中间层、投影层等不需要 BN 的场景
    """
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False ):
        super().__init__()
        # ⚠️ 修改点 1: 增加了 bias 参数并默认为 True。
        # 在没有 BN 的情况下，卷积层通常需要偏置(bias)来学习截距，否则拟合能力会受限。
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        
        # ⚠️ 修改点 2: 激活函数逻辑优化，兼容 YOLO11 写法
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))   


class SCAM(nn.Module):
    """
    Spatial-Channel Attention Module (SCAM) for YOLO11
    """
    def __init__(self, c1, c2, reduction=1):
        """
        初始化 SCAM 模块
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数 (注意：SCAM 通常保持通道不变，即 c1 == c2)
            reduction (int): 缩减比率 (逻辑保留，但当前代码强制 inter_channels = c1)
        """
        super().__init__()
        # 为了保证 x + y 能相加，输入输出通道必须一致
        # 虽然 YOLO 解析会传 c2 进来，但我们强制使用 c1 作为核心维度
        self.in_channels = c1
        self.inter_channels = c1  # 保持原逻辑：self.inter_channels = in_channels

        # --- 定义子模块 ---
        
        # k: 生成 Key 向量 [N, 1, H, W] -> 用于空间 Softmax
        # 对应原代码: self.k = Conv(in_channels, 1, 1, 1)
        self.k = Conv(self.in_channels, 1, k=1, s=1, act=False)

        # v: 生成 Value 特征 [N, C, H, W]
        # 对应原代码: self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.v = Conv(self.in_channels, self.inter_channels, k=1, s=1, act=False)

        # m: 通道注意力混合 (无 BN)
        # 对应原代码: self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        # 使用 nn.Conv2d 代替，这正是 standard 1x1 conv without BN
        # self.m = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        '''  用BN后ap值反而降低  ''' 
        # self.m = Conv_withoutBN(self.inter_channels, self.in_channels, k=1, s=1)

        # m2: 空间注意力聚合 (2通道 -> 1通道)
        # 对应原代码: self.m2 = Conv(2, 1, 1, 1)
        self.m2 = Conv(2, 1, k=1, s=1)

        # 池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        """前向传播：逻辑与原版完全一致"""
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # ---------------- Channel Attention Branch ----------------
        # avg max: [N, C, 1, 1] -> [N, 1, 1, C]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # ---------------- Spatial Attention Branch ----------------
        # k: [N, 1, H, W] -> [N, 1, HW, 1] -> Softmax over spatial
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, C, H, W] -> [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # ---------------- Fusion ----------------
        # y: [N, 1, C, 1] -> [N, C, 1, 1] (Global Channel Context)
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y_avg / y_max: [N, 1, 1, HW] -> [N, 1, H, W] (Spatial Maps)
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat: [N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        # Final Computation: Channel Attn * Spatial Attn Mask
        # self.m(y): [N, C, 1, 1]
        # self.m2(y_cat): [N, 1, H, W]
        # Broadcasting happens here
        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y
    

# class SCAM1(nn.Module):
#     """
#     Spatial-Channel Attention Module (SCAM) for YOLO11
#     """
#     def __init__(self, c1, c2, reduction=1):
#         """
#         初始化 SCAM 模块
#         Args:
#             c1 (int): 输入通道数
#             c2 (int): 输出通道数 (注意：SCAM 通常保持通道不变，即 c1 == c2)
#             reduction (int): 缩减比率 (逻辑保留，但当前代码强制 inter_channels = c1)
#         """
#         super().__init__()
#         # 为了保证 x + y 能相加，输入输出通道必须一致
#         # 虽然 YOLO 解析会传 c2 进来，但我们强制使用 c1 作为核心维度
#         self.in_channels = c1
#         self.inter_channels = c1  # 保持原逻辑：self.inter_channels = in_channels

#         # --- 定义子模块 ---
        
#         # k: 生成 Key 向量 [N, 1, H, W] -> 用于空间 Softmax
#         # 对应原代码: self.k = Conv(in_channels, 1, 1, 1)
#         self.k = Conv(self.in_channels, 1, k=1, s=1)

#         # v: 生成 Value 特征 [N, C, H, W]
#         # 对应原代码: self.v = Conv(in_channels, self.inter_channels, 1, 1)
#         self.v = Conv(self.in_channels, self.inter_channels, k=1, s=1)

#         # m: 通道注意力混合 (无 BN)
#         # 对应原代码: self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
#         # 使用 nn.Conv2d 代替，这正是 standard 1x1 conv without BN
#         # self.m = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        
#         # 
#         self.m = Conv_withoutBN2(self.inter_channels, self.in_channels, k=1, s=1)

#         # m2: 空间注意力聚合 (2通道 -> 1通道)
#         # 对应原代码: self.m2 = Conv(2, 1, 1, 1)
#         self.m2 = Conv(2, 1, k=1, s=1)

#         # 池化层
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#     def forward(self, x):
#         """前向传播：逻辑与原版完全一致"""
#         n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

#         # ---------------- Channel Attention Branch ----------------
#         # avg max: [N, C, 1, 1] -> [N, 1, 1, C]
#         avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
#         max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

#         # ---------------- Spatial Attention Branch ----------------
#         # k: [N, 1, H, W] -> [N, 1, HW, 1] -> Softmax over spatial
#         k = self.k(x).view(n, 1, -1, 1).softmax(2)

#         # v: [N, C, H, W] -> [N, 1, C, HW]
#         v = self.v(x).view(n, 1, c, -1)

#         # ---------------- Fusion ----------------
#         # y: [N, 1, C, 1] -> [N, C, 1, 1] (Global Channel Context)
#         y = torch.matmul(v, k).view(n, c, 1, 1)

#         # y_avg / y_max: [N, 1, 1, HW] -> [N, 1, H, W] (Spatial Maps)
#         y_avg = torch.matmul(avg, v).view(n, 1, h, w)
#         y_max = torch.matmul(max, v).view(n, 1, h, w)

#         # y_cat: [N, 2, H, W]
#         y_cat = torch.cat((y_avg, y_max), 1)

#         # Final Computation: Channel Attn * Spatial Attn Mask
#         # self.m(y): [N, C, 1, 1]
#         # self.m2(y_cat): [N, 1, H, W]
#         # Broadcasting happens here
#         y = self.m(y) * self.m2(y_cat).sigmoid()

#         return x + y
    


"""
DsP-YOLO：LCBHAM模块
"""



class LCAM(nn.Module):
    """
    Lightweight Channel Attention Module (LCAM)
    论文 4.2.1 节: 通道注意力路径
    特点: 使用 1x1 卷积代替全连接层，压缩比 r=16
    """
    def __init__(self, c1, ratio=16):
        super().__init__()
        # 降维后的通道数，最小为1防止报错
        c_hidden = max(1, c1 // ratio)
        
        # 共享的多层感知机 (Shared MLP)，用 1x1 卷积实现
        # 论文公式 (3): Conv(Relu(Conv(x)))
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c_hidden, c1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 全局平均池化 [cite: 310]
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        # 2. 全局最大池化 [cite: 305]
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        # 3. 相加并激活 [cite: 313]
        attention = self.sigmoid(avg_out + max_out)
        return attention

class LD_SAM(nn.Module):
    """
    Lightweight Detail-Sensitive Spatial Attention Module (LD-SAM)
    论文 4.2.1 节: 空间注意力路径
    特点: 使用 3x3 卷积代替 7x7 卷积，对小目标细节更敏感
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        # 论文图 4 LD-SAM 部分: Max(dim=1) 和 Mean(dim=1) 拼接后是 2 通道 [cite: 315, 321]
        # 卷积核必须是 3x3 [cite: 319]
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=autopad(kernel_size), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿通道维度的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 2. 拼接 [cite: 337]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 3. 3x3 卷积 + Sigmoid [cite: 339]
        attention = self.sigmoid(self.conv(x_cat))
        return attention

class LCBHAM(nn.Module):
    """
    Lightweight Channel Block Hardswish Attention Module (LCBHAM)
    论文主要创新模块: 替换 PAN 底层的 CBS
    结构: Conv(3x3, s=2) -> BN -> Hardswish -> LCAM -> LD-SAM
    """
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1):
        super().__init__()
        # 1. 特征变换模块 [cite: 302, 303, 304]
        # 注意：这里默认 s=2 (stride=2)，因为它是用来做下采样的
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k//2 if p is None else p, groups=g, dilation=d, bias=False),
            nn.BatchNorm2d(c2),
            nn.Hardswish(inplace=True) # 论文强调使用 Hardswish [cite: 264]
        )
        
        # 2. 嵌入的注意力模块
        self.lcam = LCAM(c2)    # 通道注意力
        self.ld_sam = LD_SAM()  # 空间注意力

    '''方案2：直接串行'''
    # def forward(self, x):
    #     # 第一步：特征变换 (Conv+BN+Hardswish)
    #     x = self.conv(x)
    #     # 第二步：通道注意力增强 [cite: 299]
    #     x = x * self.lcam(x)
    #     # 第三步：空间注意力增强 (论文图4显示是串行的)
    #     x = x * self.ld_sam(x)
    #     return x
    
    '''方案1：双分支门控机制'''
    def forward(self, x):
        # 0. 基础特征提取 (得到 F_org)
        f_org = self.conv(x)    
        
        # 1. 通道增强分支 [对应图 4 第一个乘法节点]
        # 一条分支去计算权重 M_ch，另一条分支与其相乘
        m_ch = self.lcam(f_org)
        f_ch = f_org * m_ch   # 得到中间特征 F_ch 
        
        # 2. 空间增强分支 [对应图 4 第二个乘法节点]
        # 一条分支去计算权重 M_sp，另一条分支与其相乘
        m_sp = self.ld_sam(f_ch)
        f_out = f_ch * m_sp   # 得到最终输出 
        
        return f_out
    

# class LCBHAM1(nn.Module):
#     """
#     Lightweight Channel Block Hardswish Attention Module (LCBHAM)
#     论文主要创新模块: 替换 PAN 底层的 CBS
#     结构: Conv(3x3, s=2) -> BN -> Hardswish -> LCAM -> LD-SAM
#     """
#     def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1):
#         super().__init__()
#         # 1. 特征变换模块 [cite: 302, 303, 304]
#         # 注意：这里默认 s=2 (stride=2)，因为它是用来做下采样的
#         self.conv = nn.Sequential(
#             nn.Conv2d(c1, c2, k, s, k//2 if p is None else p, groups=g, dilation=d, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.Hardswish(inplace=True) # 论文强调使用 Hardswish [cite: 264]
#         )
        
#         # 2. 嵌入的注意力模块
#         self.lcam = LCAM(c2)    # 通道注意力
#         self.ld_sam = LD_SAM()  # 空间注意力

#     '''方案2：直接串行'''
#     def forward(self, x):
#         # 第一步：特征变换 (Conv+BN+Hardswish)
#         x = self.conv(x)
#         # 第二步：通道注意力增强 [cite: 299]
#         x = x * self.lcam(x)
#         # 第三步：空间注意力增强 (论文图4显示是串行的)
#         x = x * self.ld_sam(x)
#         return x

class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution (SPD-Conv)
    来源: "No More Strided Convolutions" (2022)
    作用: 替代 Stride=2 的卷积/池化，无损下采样，保留小目标特征。
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):

        super().__init__()
    
        # 这里的 k, s, p 是为了兼容 YAML 的传参
        # 核心逻辑不变：输入通道翻 4 倍
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, autopad(k, p), groups=g, bias=False) 
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # x shape: [B, C, H, W]
        # # 类似于 Focus 模块的切片操作
        # # 取出 (0,0), (0,1), (1,0), (1,1) 四个位置的像素
        # x0 = x[..., 0::2, 0::2] # [B, C, H/2, W/2]
        # x1 = x[..., 1::2, 0::2]
        # x2 = x[..., 0::2, 1::2]
        # x3 = x[..., 1::2, 1::2]
        
        # 在通道维度拼接
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        
        return self.act(self.bn(self.conv(x)))
    

# class SPDConv1(nn.Module):
#     """
#     v2  弃用
#     Space-to-Depth Convolution (SPD-Conv).
#     将空间维度转移到通道维度，实现无损下采样。
#     """
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super().__init__()
#         # c1: 输入通道, c2: 输出通道
#         # 下采样后通道数变为 c1 * 4，因此需要一个卷积层映射回 c2
#         self.conv = nn.Conv2d(c1 * 4, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         # 执行 space_to_depth 逻辑 (s=2 的无损下采样)
#         x = torch.cat([
#             x[..., ::2, ::2], 
#             x[..., 1::2, ::2], 
#             x[..., ::2, 1::2], 
#             x[..., 1::2, 1::2]
#         ], 1)
#         # 通过卷积层调整通道并增加非线性表达
#         return self.act(self.bn(self.conv(x)))





# ==========================================
# 1. CA / CSHA / MGFAB (保持不变)
# ==========================================
class CA(nn.Module):
    def __init__(self, in_channel, rate=4):
        super(CA, self).__init__()
        self.in_channel = in_channel
        self.mid_ch = max(1, int(in_channel / rate))
        self.cov1 = Conv(in_channel, self.mid_ch, 1)
        self.cov2 = Conv(self.mid_ch, in_channel, 1)
        self.act2 = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x_ = self.cov2(self.cov1(x))
        weight = self.act2(self.pool(x_))
        return x * weight

class CSHA(nn.Module):
    def __init__(self, in_channel=128, rate=4, hw=None):
        super(CSHA, self).__init__()
        self.ca = CA(in_channel=in_channel, rate=rate)
        self.sa = nn.Sequential(Conv(in_channel, in_channel, k=3), Conv(in_channel, in_channel, k=3))
    def forward(self, x):
        x1 = self.ca(x)
        return self.sa(x1) + x1

class MGFAB(nn.Module):
    def __init__(self, c1, c2, hw=[40, 40]):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c_half = c1 // 2 
        self.csha = CSHA(in_channel=self.c_half, rate=4, hw=hw)
        c_concat = (c1 - self.c_half) + self.c_half + self.c_half
        self.conv2 = Conv(c_concat, c2, 1)
    def forward(self, x):
        x0, x1 = torch.split(x, [self.c1 - self.c_half, self.c_half], dim=1)
        x2 = self.csha(x1)
        return self.conv2(torch.cat([x0, x1, x2], dim=1))

# ==========================================
# 2. CrossAttention (关键修复: Query下采样策略)
# ==========================================
class CrossAttention(nn.Module):
    def __init__(self, c1, c2, n=1, emb_dim=256, patch_size=[8, 8], feature_size=None, dropout=0.1, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        
        # [关键修复] Query 处理: 必须与 patch_size 对齐
        # 使用 AvgPool 代替 stride conv，确保网格大小一致
        self.query_pool = nn.AvgPool2d(tuple(patch_size))
        self.query_proj = nn.Linear(c1, emb_dim, bias=False)
        
        # Key & Value 投影
        self.key = nn.Linear(c2, emb_dim, bias=False)
        self.value = nn.Linear(c2, emb_dim, bias=False)
        
        self.pool = nn.AvgPool2d(tuple(patch_size))

    def forward(self, x1, x2):
        # x1: Query (Shallow/Large), x2: Key (Deep/Small)
        b, c1, h1, w1 = x1.shape
        _, c2, h2, w2 = x2.shape
        ph, pw = self.patch_size
        num_patch_h_q, num_patch_w_q = h1 // ph, w1 // pw

        # --- 1. Key & Value (来自 x2) ---
        x_pool = self.pool(x2) # Downsample Key/Value
        x_flat = x_pool.view(b, c2, -1).transpose(1, 2) # [b, num_patch, c2]

        k = self.key(x_flat).view(b, -1, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2)
        v = self.value(x_flat).view(b, -1, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2)

        # --- 2. Query (来自 x1) [修复点] ---
        # 同样使用 pool 进行下采样，确保 patches 数量与 reshape 预期一致
        q_pool = self.query_pool(x1) 
        q_flat = q_pool.view(b, c1, -1).transpose(1, 2)
        q = self.query_proj(q_flat).view(b, -1, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2)

        # --- 3. Attention ---
        d_k = q.shape[-1]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        weights = self.dropout(torch.softmax(score, dim=-1))
        
        y = torch.matmul(weights, v) # [b, heads, num_q, d_head]
        
        # --- 4. 重组 ---
        y = y.transpose(1, 2).contiguous().view(b, num_patch_h_q, num_patch_w_q, self.emb_dim)
        y = y.permute(0, 3, 1, 2) # [b, c, h, w]
        
        # 上采样回原始分辨率
        out = torch.nn.functional.interpolate(y, size=(h1, w1), mode='bilinear', align_corners=False)
        return out

# ==========================================
# 3. CSFA (适配层)
# ==========================================
class CSFA(nn.Module):
    def __init__(self, c1, c2, n=1, hw=[40, 40], patch_size=[4, 4]):
        super().__init__()
        ch_deep, ch_shallow = c1 
        
        # MGFAB: 处理 Deep 特征
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.mgfab = MGFAB(ch_deep, ch_deep, hw=hw) 
        
        # CrossAttention: 
        # Query=Shallow(x2), Key=Deep(x1)
        # [关键设置] emb_dim = ch_deep (128)
        # 这样 Attention 的输出通道就是 128，可以和 MGFAB 拼接
        self.cross_attention = CrossAttention(
            c1=ch_shallow, c2=ch_deep, n=n, 
            emb_dim=ch_deep, # 强制输出通道为 Deep 通道数
            patch_size=patch_size, feature_size=hw
        )
        
        depth = 1 if patch_size[0] < 5 else 2
        # 输入 ch_deep, 输出 ch_deep
        self.conv3 = nn.Sequential(*[
            Conv(ch_deep, ch_deep, 3) for _ in range(depth)
        ])
        
        # 最终融合: ch_deep + ch_deep -> c2
        self.final_conv = Conv(ch_deep + ch_deep, c2, 1)

    def forward(self, x):
        x1, x2 = x # x1:Deep, x2:Shallow
        
        # Branch 1
        x1_up = self.up(x1)
        x1_out = self.mgfab(x1_up)
        
        # Branch 2 (Shallow 查询 Deep)
        x2_att = self.cross_attention(x2, x1)
        x2_out = self.conv3(x2_att) + x2_att 
        
        return self.final_conv(torch.cat([x1_out, x2_out], dim=1))
    

# ----------------- 辅助函数与基础卷积 -----------------
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])



# ----------------- 适配后的 PatchAttention -----------------
class PatchAttention(nn.Module):
    """
    改编适配 YOLO11 的 PatchAttention 模块
    Args:
        c1 (list): [c_low, c_high] 输入通道列表
        c2 (int): 输出通道 (通常等于 c_low)
        patch_size (int or list): Patch 大小，默认为 [16, 16]
        emb_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        dropout (float): Dropout 比率
    """
    def __init__(self, c1, c2, patch_size=[16, 16], emb_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        # c1 是列表 [c_low, c_high]
        if isinstance(c1, list):
            c_low, c_high = c1[0], c1[1]
        else:
            c_low, c_high = c1, c1
            
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
            
        self.num_heads = num_heads
        self.path_size = patch_size
        
        # --- 修改点：直接使用 Ultralytics 的 Conv 模块 ---
        # 这里的 Conv 包含了 Conv2d + BatchNorm + SiLU
        self.query = Conv(c_high, emb_dim, k=3, s=2) 
        # ----------------------------------------------
        
        self.key = nn.Linear(c_low, emb_dim, bias=False)
        self.pool = nn.AvgPool2d(tuple(patch_size), stride=tuple(patch_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [x_low, x_high]
        x_low, x_high_level = x[0], x[1]
        
        b, c, h, w = x_low.shape
        
        # 动态计算 Patch 数量
        patch_h_num = h // self.path_size[0]
        patch_w_num = w // self.path_size[1]

        # 1. Value 处理 (切分 Patch)
        x1 = x_low.view(b, self.num_heads, -1, h, w)
        x1 = x1.view(-1, x1.shape[2], h, w)
        patch_h_list = list(torch.split(x1, self.path_size[0], dim=2))
        patch_list = [torch.split(tensor, self.path_size[1], dim=3) for tensor in patch_h_list]
        value = torch.stack([j.flatten(start_dim=1) for i in patch_list for j in i], dim=1)

        # 2. Key 处理
        x_ = self.pool(x_low)
        x_ = x_.view(b, c, -1).transpose(1, 2)
        key = self.key(x_)
        key = transpose_qkv(key, num_heads=self.num_heads)

        # 3. Query 处理 (使用官方 Conv)
        query_feat = self.query(x_high_level) 
        query = query_feat.flatten(start_dim=2).transpose(1, 2)
        query = transpose_qkv(query, self.num_heads)

        # 4. Attention 计算
        d = query.shape[-1]
        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        weights = score.softmax(dim=-1)
        y = torch.bmm(self.dropout(weights), value)

        # 5. 重组输出
        out_patches = torch.split(y, 1, dim=1)
        out_patches = [
            patch.squeeze(1)
            .view(b, self.num_heads, -1)
            .reshape(b, self.num_heads, -1, self.path_size[0], self.path_size[1]) 
            for patch in out_patches
        ]
        out_patches = [
            patch.reshape(b, -1, self.path_size[0], self.path_size[1]) 
            for patch in out_patches
        ]
        
        temp = []
        for i in range(patch_h_num):
            row_patches = out_patches[i * patch_w_num : (i + 1) * patch_w_num]
            temp.append(torch.cat(row_patches, dim=3))
        
        out = torch.cat(temp, dim=2)
        return out