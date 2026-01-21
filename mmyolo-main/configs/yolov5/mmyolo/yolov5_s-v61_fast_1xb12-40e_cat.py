# 基于该配置进行继承并重写部分配置
_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = './data/cat/' # 数据集根路径
class_name = ('cat', ) # 数据集类别名称
num_classes = len(class_name) # 数据集类别数
# metainfo 必须要传给后面的 dataloader 配置，否则无效
# palette 是可视化时候对应类别的显示颜色
# palette 长度必须大于或等于 classes 长度
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

# 基于 tools/analysis_tools/optimize_anchors.py 自适应计算的 anchor
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]
# 最大训练 40 epoch
max_epochs = 40
# bs 为 12
train_batch_size_per_gpu = 12
# dataloader 加载进程数
train_num_workers = 4

# 加载 COCO 预训练权重
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    # 固定整个 backbone 权重，不进行训练
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)
    ))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # 数据集标注文件 json 路径
        ann_file='annotations/trainval.json',
        # 数据集前缀
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    # 每隔 10 个 epoch 保存一次权重，并且最多保存 2 个权重
    # 模型评估时候自动保存最佳模型
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # warmup_mim_iter 参数非常关键，因为 cat 数据集非常小，默认的最小 warmup_mim_iter 是 1000，导致训练过程学习率偏小
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    # 日志打印间隔为 5
    logger=dict(type='LoggerHook', interval=5))
# 评估间隔为 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])