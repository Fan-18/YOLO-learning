import numpy as np
import torch
import torch.nn as nn
import math
from nets.backbone import Backbone, C2f, Conv, CA
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.data = torch.mm(w_bn, w_conv).view(fusedconv.weight.shape)

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.data = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    return fusedconv


class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class CSHA(nn.Module):
    def __init__(self, in_channel=128, rate=4, hw=[160, 160]):
        super(CSHA, self).__init__()
        self.ca = CA(in_channel=in_channel, rate=rate)
        # self.sa = SA(in_channel=in_channel, hw=hw)
        self.sa = nn.Sequential(Conv(in_channel, in_channel, k=3), Conv(in_channel, in_channel, k=3))

    def forward(self, x):
        x1 = self.ca(x)
        return self.sa(x1) + x1


class MGFAB(nn.Module):
    def __init__(self, ch1, ch2, hw=[40, 40]):
        super().__init__()
        self.ch1 = ch1
        self.ch2 = ch2
        # self.conv0 = Conv(ch1+ch2//4, ch1, 1)
        self.csha = CSHA(ch1 // 2, hw=hw)
        # self.conv1 = Conv(ch1 // 2, ch1 // 2, 3)
        self.conv2 = Conv(3 * ch1 // 2, ch2, 1)

    def forward(self, x1):
        x1_0, x1_1 = torch.split(x1, self.ch1 // 2, dim=1)
        x1_2 = self.csha(x1_1)
        x1 = torch.cat([x1_0, x1_1, x1_2], dim=1)
        x1_out = self.conv2(x1)
        return x1_out


class PatchAttention(nn.Module):
    def __init__(self, c_low, c_high, emb_dim=256, patch_size=[16, 16], feature_size=[160, 160], dropout=0.1, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.path_size = patch_size
        self.query = Conv(c_high, emb_dim, k=3, s=2)
        self.key = nn.Linear(c_low, emb_dim, bias=False)
        self.patch_x = feature_size[0] // patch_size[0]
        self.patch_y = feature_size[1] // patch_size[1]
        self.pool_size = patch_size[0] * patch_size[1]
        self.pool = nn.AvgPool2d(tuple(patch_size))
        # self.adapter = nn.Parameter(torch.ones((self.patch_y*self.patch_x, self.pool_size, 1)))   # 100, 64, 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_high_level):       # b, c, h, w
        b, c, h, w = x.shape
        x1 = x.view(b, self.num_heads, -1, h, w)
        x1 = x1.view(-1, x1.shape[2], h, w)
        patch_h = list(torch.split(x1, self.path_size[0], dim=2))
        patch_list = [torch.split(tensor, self.path_size[1], dim=3) for tensor in patch_h]
        # patches = [j for i in patch_list for j in i]    # 100: 8b,c//8,16,16
        value = torch.stack([j.flatten(start_dim=1) for i in patch_list for j in i], dim=1)  # 8b, 100, c//8*16^2

        x_ = self.pool(x)     # b,c, 10, 10
        x_ = x_.view(b, c, -1).transpose(1, 2)  # b, 100, c
        key = self.key(x_)
        key = transpose_qkv(key, num_heads=self.num_heads)
        query = self.query(x_high_level).flatten(start_dim=2).transpose(1, 2)
        query = transpose_qkv(query, self.num_heads)
        d = query.shape[-1]
        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        weights = score.softmax(dim=-1)     # 8b ,100,100
        y = torch.bmm(self.dropout(weights), value)     # 8b, 100, c//8*16^2
        out = torch.split(y, 1, dim=1)
        out = [patch.squeeze(1).view(b, self.num_heads, -1).reshape(b, self.num_heads, -1, self.path_size[0],
                                                                    self.path_size[1]) for patch in out]
        out = [patch.reshape(b, -1, self.path_size[0], self.path_size[1]) for patch in out]
        temp = []
        for i in range(self.patch_y):
            same_y = out[i*self.patch_x:(i+1)*self.patch_x]
            temp.append(torch.cat(same_y, dim=3))   # b ,c, 8,160
        out = torch.cat(temp, dim=2)
        return out


class CrossAttention(nn.Module):
    def __init__(self, c1, c2, n, emb_dim=256, patch_size=[8, 8], feature_size=[80, 80], dropout=0.1, num_heads=8):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.path_size = patch_size
        self.query = nn.Sequential(Conv(c1, emb_dim, k=3, s=2), Conv(emb_dim, emb_dim, k=3, s=2)) if n > 1 else\
                     Conv(c1, emb_dim, k=3, s=2)
        self.key = nn.Linear(c2, emb_dim, bias=False)
        self.patch_x = feature_size[0] // patch_size[0]
        self.patch_y = feature_size[1] // patch_size[1]
        self.pool_size = patch_size[0] * patch_size[1]
        self.pool = nn.AvgPool2d(tuple(patch_size))
        # self.learnable_pool = nn.Parameter(torch.rand((self.patch_y*self.patch_x, self.pool_size, 1)))   # 100, 64, 1

    def forward(self, x1, x2):       # b, c, h, w
        """
        x1: 高级语义作 Q
        x2: 低级特征，作 K,V
        """
        b, c, h, w = x2.shape
        x2_ = x2.reshape(b, self.num_heads, -1, h, w)
        x2_ = x2_.reshape(-1, x2_.shape[2], h, w)   # 8b, c//8, h, w
        patch_h = list(torch.split(x2_, self.path_size[0], dim=2))
        patch_list = [torch.split(tensor, self.path_size[1], dim=3) for tensor in patch_h]
        # patches = [j for i in patch_list for j in i]    # 100: 8b, c//8, 8, 8
        value = torch.stack([j.flatten(start_dim=1) for i in patch_list for j in i], dim=1)  # 8b, 100, c//8*8*8

        x_ = self.pool(x2)
        x_ = x_.view(b, c, -1).transpose(1, 2)  # b, 100, c
        key = self.key(x_)
        key = transpose_qkv(key, self.num_heads)
        query = self.query(x1).flatten(start_dim=2).transpose(1, 2)
        query = transpose_qkv(query, self.num_heads)
        d = query.shape[-1]
        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        weights = score.softmax(dim=-1)     # 8b ,100,100
        y = torch.bmm(self.dropout(weights), value)
        out = torch.split(y, 1, dim=1)
        out = [patch.squeeze(1).view(b, self.num_heads, -1).reshape(b, self.num_heads, -1, self.path_size[0], self.path_size[1]) for patch in out]
        out = [patch.reshape(b, -1, self.path_size[0], self.path_size[1]) for patch in out]
        temp = []
        for i in range(self.patch_y):
            same_y = out[i*self.patch_x:(i+1)*self.patch_x]
            temp.append(torch.cat(same_y, dim=3))   # b ,c, 8,80
        out = torch.cat(temp, dim=2)
        return out


class CSFAProV2(nn.Module):
    def __init__(self, ch1, ch2, n=1, hw=[40, 40], patch_size=[4, 4]):
        super(CSFAProV2, self).__init__()
        self.ch1 = ch1
        self.ch2 = ch2
        # self.conv0 = Conv(ch1+ch2//4, ch1, 1)
        self.mgfab = MGFAB(ch1, ch1)
        self.pooling = nn.AvgPool2d(kernel_size=tuple(patch_size), stride=tuple(patch_size))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.patchattention = PatchAttention(self.ch2_out, patch_size=patch_size, feature_size=hw)
        self.cross_attention = CrossAttention(c1=ch1, c2=ch2, n=n, patch_size=patch_size, feature_size=hw)
        self.conv3 = nn.Sequential(*[Conv(self.ch2, self.ch2, 3) for _ in range(1 if patch_size[0] < 5 else 2)])

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        x2 = self.cross_attention(x1, x2)
        x2_out = self.conv3(x2) + x2

        x1_out = self.mgfab(x1_up)
        return torch.cat([x1_out, x2_out], dim=1)


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        hw1 = [i//16 for i in input_shape]
        ps1 = [i//10 for i in hw1]
        hw2 = [i//8 for i in input_shape]
        ps2 = [i // 10 for i in hw2]
        hw3 = [i//4 for i in input_shape]
        ps3 = [i // 10 for i in hw3]

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        # ---------------------------------------------------#
        self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.adapter = Conv(int(base_channels * 16 * deep_mul), base_channels * 16, k=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # ------------------------加强特征提取网络------------------------#
        self.CSFA1 = CSFAProV2(ch1=base_channels * 16, ch2=base_channels * 8, n=1, hw=hw1, patch_size=ps1)
        self.conv_for_csfa1 = Conv(base_channels * 24, base_channels * 16, k=3)
        self.CSFA2 = CSFAProV2(ch1=base_channels * 8, ch2=base_channels * 4, n=2, hw=hw2, patch_size=ps2)
        self.conv_for_csfa2 = Conv(base_channels * 12, base_channels * 8, k=3)
        self.CSFA3 = CSFAProV2(ch1=base_channels * 16, ch2=base_channels * 8, n=2, hw=hw2, patch_size=ps2)
        self.conv_for_csfa3 = Conv(base_channels * 24, base_channels * 16, k=3)

        self.conv_for_cat1 = Conv(base_channels * 6, base_channels * 4, k=3)
        self.conv_for_cat2 = Conv(base_channels * 12, base_channels * 8, k=3)
        self.conv_for_cat3 = Conv(base_channels * 24, base_channels * 16, k=3)

        self.pa = PatchAttention(c_low=2 * base_channels, c_high=16 * base_channels, patch_size=ps3, feature_size=hw3)
        self.conv_for_pa = Conv(2 * base_channels, 2 * base_channels, k=7)
        self.conv_all = Conv(base_channels * 30, base_channels * 30, k=1)
        # ------------------------加强特征提取网络------------------------#

        ch = [base_channels * 30]
        self.shape = None
        self.nl = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride = torch.tensor([4])  # forward
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, x):
        #  backbone
        feat0, feat1, feat2, feat3 = self.backbone.forward(x)
        feat3 = self.adapter(feat3)

        # ------------------------加强特征提取网络------------------------#
        c5_1 = self.CSFA1(feat3, feat2)
        c5_1 = self.conv_for_csfa1(c5_1)
        c4_1 = self.CSFA2(feat2, feat1)
        c4_1 = self.conv_for_csfa2(c4_1)
        f1_up = self.up(feat1)
        c3_1 = self.conv_for_cat1(torch.cat([feat0, f1_up], dim=1))
        c5_2 = self.CSFA3(c5_1, c4_1)
        c5_2 = self.conv_for_csfa3(c5_2)
        c4_1_up = self.up(c4_1)
        c4_2 = self.conv_for_cat2(torch.cat([c3_1, c4_1_up], dim=1))
        c5_2_up = self.up(c5_2)
        c5_3 = self.conv_for_cat3(torch.cat([c4_2, c5_2_up], dim=1))
        c2 = self.pa(feat0, feat3)
        c2 = self.conv_for_pa(c2)
        Pout = torch.cat([c2, c3_1, c4_2, c5_3], dim=1)  # 160, 160, 960
        Pout = self.conv_all(Pout)

        # ------------------------加强特征提取网络------------------------#
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = Pout.shape  # BCHW

        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [Pout]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400;
        #                                           box self.reg_max * 4, 8400
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


if __name__=="__main__":
    from torchinfo import summary
    base_channels = 32
    model = MGFAB(int(base_channels * 12.5), base_channels * 8)
    print(summary(model, (1, int(base_channels * 12.5), 40, 40)))
