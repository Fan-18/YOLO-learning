# YOLOv5-X P2 VisDrone 修正版配置
_base_ = './mmyolo/yolov5_x-v61_syncbn_fast_8xb16-300e_coco.py'

# ======================== 1. 模型核心参数 (保持 P2 结构) ========================
num_classes = 10
img_scale = (1024, 1024)
deepen_factor = 1.33 
widen_factor = 1.25  

model = dict(
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # last_stage_out_channels=1024,
        out_indices=(1, 2, 3, 4), # 确保输出 P2(strides 4), P3(8), P4(16), P5(32)
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='SiLU', inplace=True)),
    
    neck=dict(
        type='YOLOv5PAFPN', # 使用改进的 PAFPN 结构
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[128, 256, 512, 1024], # 对应 P2, P3, P4, P5 的输入通道
        out_channels=[128, 256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='SiLU', inplace=True),
        # 关键修改：增加一次上采样和下采样连接
        # 标准 YOLOv5 只有 2 次融合，P2 版本需要 3 次
        # 注意：这里需要确保使用的 YOLOv5PAFPN 逻辑支持 4 层输入
        # 如果是 MMYOLO，它会自动根据 in_channels 的长度调整路径
    ),

    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=10,
            in_channels=[128, 256, 512, 1024], # 对应 4 个检测头的输入通道
            widen_factor=widen_factor,
            featmap_strides=[4, 8, 16, 32]), # 严格对应 P2-P5 的步长
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [(5, 6), (8, 14), (15, 11)],      # P2 (针对超小目标)
                [(10, 13), (16, 30), (33, 23)],   # P3
                [(30, 61), (62, 45), (59, 119)],  # P4
                [(116, 90), (156, 198), (373, 326)] # P5
            ],
            strides=[4, 8, 16, 32]),
        obj_level_weights=[4.0, 1.0, 0.4, 0.1])) # 增加浅层特征图的损失权重

# ======================== 2. 数据 Pipeline (复刻 Baseline 并修复 pad_param) ========================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'))
]

# ======================== 3. 数据集与 Dataloader (采用最简复刻) ========================
data_root = 'data/visdrone/' 
class_name = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
metainfo = {
    'classes': ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 
                'tricycle', 'awning-tricycle', 'bus', 'motor'),
    'palette': [ # 给不同类别分配颜色，方便观察
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)
    ]
}

train_dataloader = dict(
    batch_size=6, 
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='train_images/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True, # 开启内存锁，加快速度
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img='val_images/'),
        test_mode=True,
        pipeline=test_pipeline,
        batch_shapes_cfg=None)) # P2 小目标检测建议关闭此项以保持一致性

test_dataloader = val_dataloader

# ======================== 4. 评估器 (移除多余参数) ========================
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox')
test_evaluator = val_evaluator

# ======================== 5. 训练/运行配置 ========================
# 学习率热身策略
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0001,
        begin=0,
        T_max=200,
        end=200,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=10)

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP'))

work_dir = '/home/jack/11/1027/mmyolo-main/work_dirs/yolov5_x_p2_visdrone_v2'
# load_from = '/home/jack/11/1027/mmyolo-main/work_dirs/yolov5_x_p2_visdrone_experiment_v1/epoch_300.pth'
resume = False