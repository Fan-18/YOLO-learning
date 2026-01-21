# beseline

# _base_ = './mmyolo/yolov5_s-v61_fast_1xb12-40e_cat.py'
_base_ = './mmyolo/yolov5_x-v61_syncbn_fast_8xb16-300e_coco.py'

# 1. 类别修改：VisDrone 是 10 类
num_classes = 10
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)
    )
)

# 2. 数据路径与格式配置
data_root = 'data/visdrone/'
class_name = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

metainfo = dict(classes=class_name)

train_dataloader = dict(
    batch_size=16, # 5090 显存大，可以调到 32 或更高
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train_images/'),
        metainfo=metainfo))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val_images/'),
        metainfo=metainfo))

test_dataloader = val_dataloader

# 3. 针对小目标的论文级改进：调大输入分辨率
# 原本是 640，我们调到 1024
img_scale = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LetterResize', scale=img_scale, allow_any_scale=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 覆盖验证和测试时的评估器配置
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'val.json', 
    metric='bbox',
    format_only=False
    # 删掉 metainfo=metainfo 这行
)

test_evaluator = val_evaluator