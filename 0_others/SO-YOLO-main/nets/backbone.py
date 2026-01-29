import math

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DConvEnhance(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=[1, 2, 3], act=True):
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s, p, g, d[0], act)
        self.conv2 = Conv(c1, c2, k, s, p, g, d[1], act)
        self.conv3 = Conv(c1, c2, k, s, p, g, d[2], act)

    def forward(self, x):
        return self.conv1(x)+self.conv2(x)+self.conv3(x)


class PatchAttention(nn.Module):
    def __init__(self, c1, emb_dim=256, patch_size=[8, 8], feature_size=[160, 160]):
        super().__init__()
        self.emb_dim = emb_dim
        self.path_size = patch_size
        self.query = nn.Linear(c1, emb_dim)
        self.key = nn.Linear(c1, emb_dim)
        self.patch_x = feature_size[0] // patch_size[0]
        self.patch_y = feature_size[1] // patch_size[1]
        self.pool = nn.ModuleList(nn.Linear(patch_size[0]*patch_size[1], 1, bias=False) for _ in range(self.patch_x * self.patch_y))
        self.conv = Conv(c1, c1, 3)
        self.relu = nn.ReLU()
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=emb_dim,
        #     nhead=8,
        #     dim_feedforward=1024,
        #     dropout=0.1,
        #     activation='relu'
        # )
        # self.norm = nn.LayerNorm(emb_dim)
        # self.encode = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)
        # self.position_encode = nn.Parameter(torch.zeros(100, emb_dim))
        # self.up = nn.PixelShuffle(upscale_factor=8)
        self.init_weight()

    def init_weight(self):
        for layer in self.pool:
            nn.init.kaiming_normal_(layer.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):       # b, c, h, w
        b, c, h, w = x.shape
        patch_h = list(torch.split(x, self.path_size[1], dim=2))
        patch_list = [torch.split(tensor, self.path_size[0], dim=3) for tensor in patch_h]
        patches = [j for i in patch_list for j in i]
        seq = []
        for i in range(len(patches)):
            seq.append(self.relu(self.pool[i](patches[i].flatten(start_dim=2))))
        seq_x = torch.cat(seq, dim=2)
        x_ = seq_x.transpose(1, 2)
        query = self.query(x_)
        key = self.key(x_)
        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(c)
        weights = score.softmax(dim=-1)     # b ,400,400
        y = torch.stack([j.flatten(start_dim=1) for j in patches], dim=-1)  # b, c88, 400
        y = y.transpose(1, 2)
        y = torch.bmm(weights, y)
        out = torch.split(y, 1, dim=1)
        out = [patch.squeeze(1).reshape(b, c, self.path_size[1], self.path_size[0]) for patch in out]
        temp = []
        for i in range(self.patch_y):
            same_y = out[i*self.patch_x:(i+1)*self.patch_x]
            temp.append(torch.cat(same_y, dim=3))   # b ,c, 8,160
        out = torch.cat(temp, dim=2)
        out = self.conv(out)
        return out


class CA(nn.Module):
    def __init__(self, in_channel, rate=4,):
        super(CA, self).__init__()
        self.in_channel = in_channel
        self.mid_ch = int(in_channel / rate)
        self.cov1 = Conv(in_channel, self.mid_ch, 1)
        self.cov2 = Conv(self.mid_ch, in_channel, 1)
        self.act2 = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_ = self.cov2(self.cov1(x))
        weight = self.act2(self.pool(x_))
        out = x * weight
        return out


class SA(nn.Module):
    def __init__(self, in_channel, hw=[160, 160], ):
        super().__init__()
        self.h, self.w = hw
        self.inchannel = in_channel
        self.fc_w = nn.ModuleList([nn.Sequential(nn.Linear(self.w, 1),
                                   nn.BatchNorm1d(self.h, momentum=0.05, affine=True, track_running_stats=True)) for _ in range(in_channel)])
        self.fc_h = nn.ModuleList([nn.Sequential(nn.Linear(self.h, 1),
                                   nn.BatchNorm1d(self.w, momentum=0.05, affine=True, track_running_stats=True)) for _ in range(in_channel)])
        self.act2 = nn.Softmax(dim=1)

        self.__init__weight()

    def __init__weight(self):
        for c in range(self.inchannel):
            nn.init.kaiming_normal_(self.fc_w[c][0].weight.data, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc_h[c][0].weight.data, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.fc_w[c][0].bias.data, 0.0)
            nn.init.constant_(self.fc_h[c][0].bias.data, 0.0)
            nn.init.normal_(self.fc_w[c][1].weight.data, 1.0, 0.02)
            nn.init.normal_(self.fc_h[c][1].weight.data, 1.0, 0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        assert self.h == h and self.w == w, "feature's h and w must match hw param!"
        out_list = []
        x = x.permute(1, 0, 2, 3)
        for c in range(self.inchannel):
            h_weight = self.act2(self.fc_w[c](x[c]))  # b,h,1
            h_weight = h_weight.permute(0, 2, 1).repeat(1, self.h, 1)  # b,h,h
            x_c = torch.bmm(h_weight, x[c])
            x_c = x_c.permute(0, 2, 1)  # b, w, h
            w_weight = self.act2(self.fc_h[c](x_c))
            w_weight = w_weight.permute(0, 2, 1).repeat(1, self.w, 1)  # b, w,w
            x_out = torch.bmm(w_weight, x_c).permute(0, 2, 1)  # b ,h,w
            out_list.append(x_out)
        output = torch.stack(out_list, dim=0).permute(1, 0, 2, 3)
        return output


class SA1(nn.Module):
    def __init__(self, in_channel, hw=[160, 160], ):
        super().__init__()
        self.h, self.w = hw
        self.inchannel = in_channel
        self.fc_w = nn.Linear(self.w, 1)
        self.fc_h = nn.Linear(self.h, 1)

        self.act2 = nn.Softmax(dim=1)

        self.__init__weight()

    def __init__weight(self):
        nn.init.kaiming_normal_(self.fc_w.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_h.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc_w.bias.data, 0.0)
        nn.init.constant_(self.fc_h.bias.data, 0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert self.h == h and self.w == w, "feature's h and w must match hw param!"

        x = x.view(-1, h, w)        # bc, h, w
        h_weight = self.act2(self.fc_w(x))  # bc,h,1
        h_weight = h_weight.permute(0, 2, 1).repeat(1, self.h, 1)  # bc,h,h
        x_c = torch.bmm(h_weight, x)
        x_c = x_c.permute(0, 2, 1)  # bc, w, h
        w_weight = self.act2(self.fc_h(x_c))
        w_weight = w_weight.permute(0, 2, 1).repeat(1, self.w, 1)  # bc, w,w
        x_out = torch.bmm(w_weight, x_c).permute(0, 2, 1)  # bc, h, w
        output = x_out.view(b, c, h, w)
        return output


class CSHA(nn.Module):
    def __init__(self, in_channel=128, rate=4, hw=[160, 160]):
        super(CSHA, self).__init__()
        self.ca = CA(in_channel=in_channel, rate=rate)
        # self.sa = SA(in_channel=in_channel, hw=hw)
        self.sa = nn.Sequential(Conv(in_channel, in_channel, k=7), Conv(in_channel, in_channel, 7))

    def forward(self, x):
        x1 = self.ca(x)
        return self.sa(x1) + x1


class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    


class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        feat0 = x
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return  feat0, feat1, feat2, feat3


if __name__=="__main__":
    x = torch.rand((1, 64, 160, 160))
    model = PatchAttention(64, emb_dim=64)
    # import time
    # t1 = time.time()
    # y = model(x)
    # t2 = time.time()
    # print(t2-t1)
    from torchinfo import summary

    model2 = CSHA(128, hw=[80,80])
    model3 = DConvEnhance(128, 128)
    print(summary(model3, (1, 128, 160, 160)))
