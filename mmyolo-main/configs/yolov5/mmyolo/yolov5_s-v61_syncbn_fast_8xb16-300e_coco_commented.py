# ======================== 基础配置 ========================
# 后端参数
_backend_args = None

# 多尺度测试时的图像预处理变换（包含三种尺度：320x320, 640x640, 960x960）
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]

# ======================== 数据增强配置 ========================
# 仿射变换的缩放系数
affine_scale = 0.5

# Albumentations库的图像增强变换
albu_train_transforms = [
    dict(p=0.01, type='Blur'),  # 模糊
    dict(p=0.01, type='MedianBlur'),  # 中值滤波
    dict(p=0.01, type='ToGray'),  # 灰度化
    dict(p=0.01, type='CLAHE'),  # 对比度受限自适应直方图均衡化
]

# ======================== Anchor配置 ========================
# 先验框（anchor）配置，三个检测层分别对应不同的anchor尺寸
# P3层（8x步长）、P4层（16x步长）、P5层（32x步长）
anchors = [
    [
        (
            10,
            13,
        ),
        (
            16,
            30,
        ),
        (
            33,
            23,
        ),
    ],
    [
        (
            30,
            61,
        ),
        (
            62,
            45,
        ),
        (
            59,
            119,
        ),
    ],
    [
        (
            116,
            90,
        ),
        (
            156,
            198,
        ),
        (
            373,
            326,
        ),
    ],
]

# ======================== 后端和学习率配置 ========================
backend_args = None

# 基础学习率
base_lr = 0.01

# ======================== 批处理配置 ========================
# 动态批处理形状配置，用于自适应调整不同大小的图像
batch_shapes_cfg = dict(
    batch_size=1,
    extra_pad_ratio=0.5,
    img_size=640,
    size_divisor=32,
    type='BatchShapePolicy')

# ======================== 自定义钩子 ========================
# 自定义钩子配置（包括EMA指数移动平均）
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',  # 指数移动平均类型
        momentum=0.0001,  # EMA的动量系数
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]

# ======================== 数据集配置 ========================
# COCO数据集根目录
data_root = 'data/coco/'

# 数据集类型
dataset_type = 'YOLOv5CocoDataset'

# ======================== 模型缩放因子 ========================
# 模型深度缩放因子（0.33表示使用33%的深度，YOLOv5s配置）
deepen_factor = 0.33

# ======================== 默认钩子配置 ========================
default_hooks = dict(
    # 检查点保存配置：每10个epoch保存，保留最新3个
    checkpoint=dict(
        interval=10, max_keep_ckpts=3, save_best='auto',
        type='CheckpointHook'),
    # 日志记录配置：每50次迭代记录一次
    logger=dict(interval=50, type='LoggerHook'),
    # 参数调度器配置：线性学习率调度
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=300,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    # 分布式采样器种子钩子（用于多GPU训练时的数据同步）
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 迭代计时器
    timer=dict(type='IterTimerHook'),
    # 检测结果可视化钩子
    visualization=dict(type='mmdet.DetVisualizationHook'))

# 默认作用域
default_scope = 'mmyolo'

# ======================== 环境配置 ========================
env_cfg = dict(
    cudnn_benchmark=True,  # 启用CUDNN自动调优
    dist_cfg=dict(backend='nccl'),  # 分布式训练后端
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))  # 多进程配置

# ======================== 图像尺寸配置 ========================
# 基础输入图像尺寸
img_scale = (
    640,
    640,
)

# TTA（测试时增强）用的多个图像尺度
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]

# ======================== 模型加载和日志配置 ========================
# 预训练模型加载路径（None表示从头开始训练）
load_from = None

# 日志级别
log_level = 'INFO'

# 日志处理器配置（按epoch统计，窗口大小50次迭代）
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

# ======================== 损失函数权重配置 ========================
# 边界框回归损失权重
loss_bbox_weight = 0.05

# 分类损失权重
loss_cls_weight = 0.5

# 目标性（objectness）损失权重
loss_obj_weight = 1.0

# ======================== 学习率和训练周期配置 ========================
# 学习率衰减因子
lr_factor = 0.01

# 最大训练轮数
max_epochs = 300

# 最多保留的检查点数
max_keep_ckpts = 3

# ======================== 模型配置 ========================
model = dict(
    # 主干网络（backbone）：YOLOv5的改进Darknet
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),  # 激活函数
        deepen_factor=0.33,  # 深度缩放因子
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),  # 批归一化配置
        type='YOLOv5CSPDarknet',  # YOLOv5 CSP Darknet骨干网
        widen_factor=0.5),  # 宽度缩放因子

    # 检测头配置
    bbox_head=dict(
        head_module=dict(
            featmap_strides=[  # 特征图步长（P3、P4、P5层）
                8,
                16,
                32,
            ],
            in_channels=[  # 输入通道数
                256,
                512,
                1024,
            ],
            num_base_priors=3,  # 每个位置的基础anchor数量
            num_classes=80,  # COCO数据集类别数
            type='YOLOv5HeadModule',
            widen_factor=0.5),  # 宽度缩放因子

        # 边界框损失函数配置
        loss_bbox=dict(
            bbox_format='xywh',  # 边界框格式（中心坐标+宽高）
            eps=1e-07,
            iou_mode='ciou',  # 使用CIoU损失
            loss_weight=0.05,
            reduction='mean',
            return_iou=True,
            type='IoULoss'),

        # 分类损失函数配置
        loss_cls=dict(
            loss_weight=0.5,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),  # 使用sigmoid激活进行多标签分类

        # 目标性（objectness）损失函数配置
        loss_obj=dict(
            loss_weight=1.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),

        # 不同特征层的目标性损失权重
        obj_level_weights=[
            4.0,  # 小物体层（P3层）
            1.0,  # 中物体层（P4层）
            0.4,  # 大物体层（P5层）
        ],

        # 先验框生成器配置
        prior_generator=dict(
            base_sizes=[
                [
                    (
                        10,
                        13,
                    ),
                    (
                        16,
                        30,
                    ),
                    (
                        33,
                        23,
                    ),
                ],
                [
                    (
                        30,
                        61,
                    ),
                    (
                        62,
                        45,
                    ),
                    (
                        59,
                        119,
                    ),
                ],
                [
                    (
                        116,
                        90,
                    ),
                    (
                        156,
                        198,
                    ),
                    (
                        373,
                        326,
                    ),
                ],
            ],
            strides=[
                8,
                16,
                32,
            ],
            type='mmdet.YOLOAnchorGenerator'),
        prior_match_thr=4.0,  # 先验框匹配阈值
        type='YOLOv5Head'),

    # 数据预处理器配置
    data_preprocessor=dict(
        bgr_to_rgb=True,  # 将BGR格式转换为RGB格式
        mean=[  # 图像均值（YOLOv5不使用标准化）
            0.0,
            0.0,
            0.0,
        ],
        std=[  # 图像标准差（直接除以255进行归一化）
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),

    # 颈部网络（Neck）：PAFPN特征融合
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),  # 激活函数
        deepen_factor=0.33,  # 深度缩放因子
        in_channels=[  # 输入通道数
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),  # 批归一化配置
        num_csp_blocks=3,  # CSP块数量
        out_channels=[  # 输出通道数
            256,
            512,
            1024,
        ],
        type='YOLOv5PAFPN',  # 路径聚合特征金字塔网络
        widen_factor=0.5),  # 宽度缩放因子

    # 测试配置
    test_cfg=dict(
        max_per_img=300,  # 每张图像最多保留的检测框数
        multi_label=True,  # 是否支持多标签
        nms=dict(iou_threshold=0.65, type='nms'),  # 非极大值抑制配置
        nms_pre=30000,  # NMS前保留的检测框数
        score_thr=0.001),  # 分数阈值
    type='YOLODetector')  # 检测器类型

# ======================== 模型测试配置 ========================
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.65, type='nms'),
    nms_pre=30000,
    score_thr=0.001)

# ======================== 规范化和类别配置 ========================
# 批归一化配置
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')

# 类别数（COCO数据集包含80个类别）
num_classes = 80

# 检测层数量（P3、P4、P5三层）
num_det_layers = 3

# ======================== 目标权重和优化器配置 ========================
# 不同特征层的目标性损失权重
obj_level_weights = [
    4.0,
    1.0,
    0.4,
]

# 优化器包装器配置
optim_wrapper = dict(
    constructor='YOLOv5OptimizerConstructor',  # YOLOv5专用优化器构造器
    optimizer=dict(
        batch_size_per_gpu=16,  # 每GPU批大小
        lr=0.01,  # 学习率
        momentum=0.937,  # 动量
        nesterov=True,  # 使用Nesterov加速梯度下降
        type='SGD',  # 随机梯度下降优化器
        weight_decay=0.0005),  # 权重衰减（L2正则化）
    type='OptimWrapper')

# ======================== 参数调度和其他配置 ========================
# 参数调度器
param_scheduler = None

# 持久化工作进程（加快数据加载）
persistent_workers = True

# 预处理变换
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),  # 加载图像
    dict(type='LoadAnnotations', with_bbox=True),  # 加载边界框标注
]

# 先验框匹配阈值
prior_match_thr = 4.0

# 是否从检查点恢复训练
resume = False

# ======================== 检查点和特征层配置 ========================
# 检查点保存间隔（轮数）
save_checkpoint_intervals = 10

# 特征图步长（不同尺度的特征层：P3、P4、P5）
strides = [
    8,
    16,
    32,
]

# ======================== 训练循环和评估配置 ========================
# 测试循环配置
test_cfg = dict(type='TestLoop')

# 测试数据加载器配置
test_dataloader = dict(
    batch_size=1,  # 批大小
    dataset=dict(
        ann_file='annotations/instances_val2017.json',  # 标注文件
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),  # 批处理形状策略
        data_prefix=dict(img='val2017/'),  # 数据前缀
        data_root='data/coco/',  # 数据根目录
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

# 测试评估器配置（COCO指标）
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',  # 评估指标：边界框
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')

# 测试数据处理管道
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),  # 加载图像
    dict(scale=(  # 保持宽高比缩放
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,  # 不允许放大
        pad_val=dict(img=114),  # 填充值（灰色）
        scale=(
            640,
            640,
        ),
        type='LetterResize'),  # Letter-box缩放
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]

# ======================== 训练数据集配置 ========================
# 训练数据集标注文件
train_ann_file = 'annotations/instances_train2017.json'

# 每GPU训练批大小
train_batch_size_per_gpu = 16

# 训练循环配置
train_cfg = dict(
    max_epochs=300,  # 最大训练轮数
    type='EpochBasedTrainLoop',
    val_interval=10)  # 验证间隔（每10个epoch进行一次验证）

# 训练数据前缀
train_data_prefix = 'train2017/'

# 训练数据加载器配置
train_dataloader = dict(
    batch_size=16,  # 批大小
    collate_fn=dict(type='yolov5_collate'),  # YOLOv5自定义碰撞函数
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 过滤配置
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),  # 加载图像
            dict(type='LoadAnnotations', with_bbox=True),  # 加载标注
            # Mosaic数据增强：将4张图像合成1张，增加数据多样性
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic'),
            # YOLOv5随机仿射变换（缩放、旋转、剪切等）
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_rotate_degree=0.0,  # 最大旋转角度
                max_shear_degree=0.0,   # 最大剪切角度
                scaling_ratio_range=(  # 缩放比例范围
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            # Albumentations图像增强
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                # 增强操作及其概率
                transforms=[
                    dict(p=0.01, type='Blur'),  # 模糊
                    dict(p=0.01, type='MedianBlur'),  # 中值滤波
                    dict(p=0.01, type='ToGray'),  # 灰度化
                    dict(p=0.01, type='CLAHE'),  # 对比度均衡
                ],
                type='mmdet.Albu'),
            # HSV颜色空间随机增强
            dict(type='YOLOv5HSVRandomAug'),
            # 随机水平翻转
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# 训练数据加载线程数
train_num_workers = 8

# 训练数据处理管道
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]

# ======================== 测试时增强（TTA）配置 ========================
# 测试时增强（TTA）模型配置
tta_model = dict(
    tta_cfg=dict(
        max_per_img=300,  # 每张图像最多保留的检测框数
        nms=dict(iou_threshold=0.65, type='nms')),  # NMS配置
    type='mmdet.DetTTAModel')

# TTA处理管道（包含多尺度和翻转）
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]

# ======================== 验证数据集配置 ========================
# 验证数据集标注文件
val_ann_file = 'annotations/instances_val2017.json'

# 每GPU验证批大小
val_batch_size_per_gpu = 1

# 验证循环配置
val_cfg = dict(type='ValLoop')

# 验证数据前缀
val_data_prefix = 'val2017/'

# 验证数据加载器配置
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

# 验证评估器配置
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')

# 验证数据加载线程数
val_num_workers = 2

# ======================== 可视化配置 ========================
# 可视化后端配置
vis_backends = [
    dict(type='LocalVisBackend'),
]

# 可视化工具配置
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),  # 本地可视化后端
    ])

# ======================== 正则化配置 ========================
# 权重衰减（L2正则化系数）
weight_decay = 0.0005

# 模型宽度缩放因子
widen_factor = 0.5
