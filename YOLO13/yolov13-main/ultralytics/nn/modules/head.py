# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
from ultralytics.nn.modules.block import SELayer, CBAM


__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "Detect_YOLOX", "Detect_UniRep", "Detect_Decoupled_SE", "Detect_Decoupled_CBAM", "Detect_SmallObj", "Detect_TaskSpecific_LK"



class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__() 
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

class Detect_YOLOX(Detect):
    """
    YOLOX Decoupled Head for comparison in Ablation Studies.
    Structure:
        Input -> 1x1 Conv (Stem) -> Split into two branches:
            Branch 1 (Cls): 3x3 Conv -> 3x3 Conv -> 1x1 Conv (Output)
            Branch 2 (Reg): 3x3 Conv -> 3x3 Conv -> 1x1 Conv (Output)
    Reference: YOLOX: Exceeding YOLO Series in 2021
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        
        # 覆盖父类 Detect 的 cv2 (Box分支) 和 cv3 (Cls分支)
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

        for x in ch:
            hidden_channels = x
            # === YOLOX Style Decoupled Head ===
            
            # 💡 进阶优化：如果是 Nano/Tiny 模型 (通道很小)，建议用 DWConv 节省参数
            # 如果通道数大于 128 (说明是 M/L/X 模型)，则用标准卷积
            ConvLayer = DWConv if hidden_channels < 128 else Conv

           # === 回归分支 (Regression) ===
            self.cv2.append(nn.Sequential(
                Conv(x, hidden_channels, 1),             # 1x1 降维/对齐
                ConvLayer(hidden_channels, hidden_channels, 3), # 3x3 卷积
                ConvLayer(hidden_channels, hidden_channels, 3), # 3x3 卷积
                nn.Conv2d(hidden_channels, 4 * self.reg_max, 1) # 输出
            ))
            
            # === 分类分支 (Classification) ===
            self.cv3.append(nn.Sequential(
                Conv(x, hidden_channels, 1),             # 1x1 降维/对齐
                ConvLayer(hidden_channels, hidden_channels, 3), 
                ConvLayer(hidden_channels, hidden_channels, 3), 
                nn.Conv2d(hidden_channels, self.nc, 1)   # 输出
            ))

# === 1. 定义坐标注意力模块 (Coordinate Attention) ===
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# === 2. 修改后的检测头 ===
class Detect_SmallObj(Detect):
    """
    针对小目标优化的检测头：
    1. 移除 DWConv,使用标准 Conv。
    2. 加入 CoordAtt 注意力机制。
    论文: Coordinate Attention for Efficient Mobile Network Design
    # 作用: 同时捕捉通道关系和长距离的位置依赖，非常适合 VisDrone 中定位密集小目标

    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        # # 重新定义 cv2 (Box) 和 cv3 (Cls)
        # c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        # --- 改进点 1: 使用标准 Conv 替代 ModuleList 生成逻辑 ---
        # --- 改进点 2: 在两个卷积之间插入 CoordAtt 注意力 ---
        
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

        for x in ch:
            # =============================================================
            # 🚀 核心改进：分层自适应通道计算
            # =============================================================
            # 1. 回归分支通道 (c2_curr)
            # 逻辑：取输入通道 x 和 4*reg_max 中的较大者，保证回归头有足够表达能力
            # 同时限制最小值为 64 (防止太窄)
            c2_curr = max((64, x, self.reg_max * 4))
            
            # 2. 分类分支通道 (c3_curr)
            # 逻辑：取输入通道 x 和 类别数 nc 中的较小者 (避免分类头过宽)
            # 同时限制最小值为 64
            c3_curr = max(64, min(self.nc, 100)) 
            # 或者简单点，直接跟随输入 x:
            # c3_curr = x 
            
            # 💡 如果你想完全跟随 model scaling (n/s/m/l)，最简单粗暴的方法是：
            # c2_curr = c3_curr = x
            
            # 这里我采用折中方案：c2_curr = x, c3_curr = x
            # 这样既能自适应，又不会引入额外的超参
            c2_curr = x
            c3_curr = x

            # === Box Branch (回归) ===
            self.cv2.append(nn.Sequential(
                Conv(x, c2_curr, 3),           # 3x3 标准卷积
                CoordAtt(c2_curr, c2_curr),    # <--- 插入 CoordAtt
                Conv(c2_curr, c2_curr, 3),     # 3x3 标准卷积
                nn.Conv2d(c2_curr, 4 * self.reg_max, 1) # Output
            ))
            
            # === Cls Branch (分类) ===
            self.cv3.append(nn.Sequential(
                Conv(x, c3_curr, 3),           # 3x3 标准卷积
                CoordAtt(c3_curr, c3_curr),    # <--- 插入 CoordAtt
                Conv(c3_curr, c3_curr, 3),     # 3x3 标准卷积
                nn.Conv2d(c3_curr, self.nc, 1) # Output
            ))

# === 2. 任务感知大感受野解耦头 (新设计) ===
class Detect_TaskSpecific_LK(Detect):
    """
    Task-Specific Large Kernel Decoupled Head (TSLK-Head).
    
    Design Philosophy:
    1. Shared Stem with CoordAtt: Locks onto small object positions early.
    2. Classification Branch: Uses 5x5 DWConv for larger context (helps distinguish small objects from background).
    3. Regression Branch: Uses 3x3 Conv for precise edge localization.
    4. Channels: Fixed to 256 (like YOLOX) to ensure sufficient feature capacity.
    1. 整体架构参考：YOLOX论文名称: YOLOX: Exceeding YOLO Series in 2021 (CVPR 2021 - Oral)借鉴点: 解耦头（Decoupled Head）。设计原理: 作者指出分类（Classification）需要平移不变性，回归（Regression）需要平移同变性。如果共享特征，会产生任务冲突（Task Misalignment）。因此必须把它们拆开。你的改进: YOLOX 的两个分支都是 $3 \times 3$ 卷积，而你改进为非对称卷积。
    2. 非对称卷积核参考：TOOD论文名称: TOOD: Task-aligned One-stage Object Detection (ICCV 2021)借鉴点: 任务对齐（Task Alignment）与 差异化感受野。设计原理: TOOD 提出，分类任务需要更多的上下文信息（Context），而定位任务关注物体边界。因此，分类分支应该拥有更大的感受野，而回归分支应该聚焦局部。你的改进: 这正是为什么我在分类分支使用了 $5 \times 5$ Depthwise Conv（大感受野），而在回归分支保留 $3 \times 3$ Conv（精准边界）的理论依据。
    3. 核心机制参考：RepLKNet / ConvNeXt论文名称: Scaling Up Your Kernels to 31x31 (CVPR 2022) 或 A ConvNet for the 2020s (CVPR 2022)借鉴点: 大核卷积（Large Kernel）与 Depthwise 结合。设计原理: 这两篇文章证明了，通过堆叠 Depthwise Convolution 扩大感受野，CNN 可以达到类似 Transformer 的效果，同时保持比 Transformer 更低的归纳偏置（Inductive Bias），非常适合提取小目标的特征。
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        # # 核心改进1：强制使用 256 通道 (对齐 YOLOX 的容量)
        # # 之前的 ch[0]//4 可能太小了，导致信息瓶颈
        # hidden_channels = 256 
        
        self.cv2 = nn.ModuleList() # Reg (Box)
        self.cv3 = nn.ModuleList() # Cls (Class)

        for x in ch:
            hidden_channels = x
            # === Shared Stem (共享主干) ===
            # 先降维/升维到 256，然后加注意力
            # 这里我们不显式定义 stem 变量，而是将其分别集成到分支入口，或者使用更高效的方法
            # 为了代码结构清晰，我们直接在分支里构建结构
            
            # === Regression Branch (回归分支 - 关注精度) ===
            # 结构: Stem(1x1) -> CoordAtt -> 3x3 -> 3x3 -> Output
            self.cv2.append(nn.Sequential(
                Conv(x, hidden_channels, 1),             # 1. 维度对齐
                CoordAtt(hidden_channels, hidden_channels), # 2. 共享位置感知 (这里每个分支独立算，效果更好但参数稍多)
                Conv(hidden_channels, hidden_channels, 3),  # 3. 标准卷积用于定位
                Conv(hidden_channels, hidden_channels, 3),  # 4. 精细调整
                nn.Conv2d(hidden_channels, 4 * self.reg_max, 1) # Output
            ))
            
            # === Classification Branch (分类分支 - 关注上下文) ===
            # 结构: Stem(1x1) -> CoordAtt -> 5x5 DWConv -> 1x1 -> Output
            # 创新点：使用 5x5 卷积扩大感受野，解决小目标语义特征弱的问题
            self.cv3.append(nn.Sequential(
                Conv(x, hidden_channels, 1),             # 1. 维度对齐
                CoordAtt(hidden_channels, hidden_channels), # 2. 位置感知
                
                # 3. 大核卷积 (Large Kernel): 5x5, padding=2
                # 使用 DWConv 节省参数量，使其不比 YOLOX 重
                DWConv(hidden_channels, hidden_channels, k=5), 
                
                # 4. 1x1 卷积混合通道信息
                Conv(hidden_channels, hidden_channels, 1),
                
                nn.Conv2d(hidden_channels, self.nc, 1)   # Output
            ))


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))



class Detect_Decoupled_SE(Detect):
    """
    Decoupled Head with Squeeze-and-Excitation (SE) Block.
    Structure: Decoupled branches with SE inserted between 3x3 convs.
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        # # 为了公平对比，内部通道数统一设为 256 (类似于 YOLOX 标准)
        # hidden_channels = 256
        
        self.cv2 = nn.ModuleList() # Regression branch
        self.cv3 = nn.ModuleList() # Classification branch

        for x in ch:
            hidden_channels = x
            # Regression Branch (回归分支)
            self.cv2.append(nn.Sequential(
                Conv(x, hidden_channels, 1),            # Stem
                Conv(hidden_channels, hidden_channels, 3), # Conv 1
                SELayer(hidden_channels),               # <--- 插入 SE Block
                Conv(hidden_channels, hidden_channels, 3), # Conv 2
                nn.Conv2d(hidden_channels, 4 * self.reg_max, 1) # Output
            ))
            
            # Classification Branch (分类分支)
            self.cv3.append(nn.Sequential(
                Conv(x, hidden_channels, 1),            # Stem
                Conv(hidden_channels, hidden_channels, 3), # Conv 1
                SELayer(hidden_channels),               # <--- 插入 SE Block
                Conv(hidden_channels, hidden_channels, 3), # Conv 2
                nn.Conv2d(hidden_channels, self.nc, 1)  # Output
            ))


class Detect_Decoupled_CBAM(Detect):
    """
    Decoupled Head with CBAM Attention.
    Structure: Decoupled branches with CBAM inserted between 3x3 convs.
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        # hidden_channels = 256
        
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

        for x in ch:
            hidden_channels = x
            # Regression Branch
            self.cv2.append(nn.Sequential(
                Conv(x, hidden_channels, 1),
                Conv(hidden_channels, hidden_channels, 3),
                CBAM(hidden_channels),                  # <--- 插入 CBAM
                Conv(hidden_channels, hidden_channels, 3),
                nn.Conv2d(hidden_channels, 4 * self.reg_max, 1)
            ))
            
            # Classification Branch
            self.cv3.append(nn.Sequential(
                Conv(x, hidden_channels, 1),
                Conv(hidden_channels, hidden_channels, 3),
                CBAM(hidden_channels),                  # <--- 插入 CBAM
                Conv(hidden_channels, hidden_channels, 3),
                nn.Conv2d(hidden_channels, self.nc, 1)
            ))

# === 1. 定义重参数化卷积块 (RepConv) ===
# 核心思想：训练是多分支，推理是单卷积。速度与精度的完美平衡。
# Ref: "RepVGG: Making VGG-style ConvNets Great Again" (CVPR 2021)
class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k
        padding = k // 2 if p is None else p
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, padding, groups=g, bias=True)
        else:
            # 训练时：三个分支 (3x3, 1x1, Identity)
            self.rbr_dense = nn.Conv2d(c1, c2, k, s, padding, groups=g, bias=False)
            self.rbr_1x1 = nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)
            self.rbr_identity = nn.BatchNorm2d(c1) if c2 == c1 and s == 1 else None
            self.bn = nn.BatchNorm2d(c2) # 简化的 BN，实际应每个分支后都有 BN

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        # 训练时的前向传播
        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
        if self.rbr_identity:
            x = x + self.rbr_identity(inputs)
        return self.act(self.bn(x))
    
    # --- 这是一个核心魔法：将多分支权重融合为一个 ---
    def fuse_repvgg_block(self):
        if self.deploy: return
        # (此处省略具体的权重合并数学公式代码，实际部署时需加上)
        # 基本原理：Kernel_final = Kernel_3x3 + Pad(Kernel_1x1) + Identity_Matrix
        # Bias_final = Bias_3x3 + Bias_1x1 + Bias_Identity
        self.deploy = True
        # ... Re-init self.rbr_reparam with fused weights ...

# === 2. 全能型重参数化检测头 ===
class Detect_UniRep(Detect):
    """
    UniRep Head: Unified Reparameterization Head.
    Inherits from 'Detect' to ensure compatibility with YOLOv8/v11 trainer.
    """
    def __init__(self, nc=80, ch=()):
        # 【关键修改1】调用父类 Detect 的初始化
        # 父类会自动处理 self.nc, self.reg_max, self.no 等属性
        super().__init__(nc, ch)
        
        # 定义内部通道数
        # max(...) 确保通道数不会太小，至少 16
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        
        # 覆盖父类的 cv2 和 cv3
        self.cv2 = nn.ModuleList() # Reg (Box)
        self.cv3 = nn.ModuleList() # Cls (Class)

        for x in ch:
            # === Reg Branch: 专注精度 ===
            # 回归任务需要精细的边缘信息，使用 3x3 卷积
            self.cv2.append(nn.Sequential(
                RepConv(x, c2, 3, 1),      # 第一层重参数化
                RepConv(c2, c2, 3, 1),     # 第二层重参数化
                nn.Conv2d(c2, 4 * self.reg_max, 1) # 输出层 (保持标准卷积)
            ))
            
            # === Cls Branch: 专注语义与大感受野 ===
            # 【关键修改2】这里直接设置 k=5 或 k=7 来实现大核
            # RepConv 会在推理时将其融合为单个 5x5 或 7x7 卷积
            # 相比 Dilated Conv，大核卷积在重参数化下更直接
            self.cv3.append(nn.Sequential(
                RepConv(x, c3, 5, 1),      # 使用 5x5 卷积获取更大感受野 (模拟 RepLKNet)
                RepConv(c3, c3, 3, 1),     # 接一个 3x3 进一步提炼
                nn.Conv2d(c3, self.nc, 1)  # 输出层
            ))

            # for x in ch:
            # # =============================================================
            # # 🚀 自适应核心：让隐藏通道直接等于输入通道 x
            # # =============================================================
            # # 为了保证 RepConv 的效果，通道数建议不要太小
            # # 这里设置一个下限 64，防止 Nano 模型在 P2 层通道过少导致 Rep 效果不佳
            # c_hidden = max(64, x) 
            
            # # 如果你想要完全跟随 YAML (即使是 32 通道)，就用:
            # # c_hidden = x

            # # === Regression Branch (回归分支) ===
            # # 两个 RepConv 串联，训练时是多分支，推理时等价于两个 3x3
            # self.cv2.append(nn.Sequential(
            #     RepConv(x, c_hidden, 3, 1),        # RepConv 1
            #     RepConv(c_hidden, c_hidden, 3, 1), # RepConv 2
            #     nn.Conv2d(c_hidden, 4 * self.reg_max, 1) # Output (Reg)
            # ))
            
            # # === Classification Branch (分类分支) ===
            # # 同样使用 RepConv 提取语义
            # self.cv3.append(nn.Sequential(
            #     RepConv(x, c_hidden, 3, 1),        # RepConv 1
            #     RepConv(c_hidden, c_hidden, 3, 1), # RepConv 2
            #     nn.Conv2d(c_hidden, self.nc, 1)    # Output (Cls)
            # ))

    def forward(self, x):
        # 标准 YOLO Detect forward 逻辑
        return [torch.cat([self.cv2[i](x[i]), self.cv3[i](x[i])], 1) for i in range(len(x))]



class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
