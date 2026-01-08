import os
import cv2
import numpy as np
import yaml
import colorsys


def load_visdrone_classes(yaml_path):
    """
    从VisDrone.yaml配置文件加载类别定义

    Args:
        yaml_path: VisDrone.yaml文件路径

    Returns:
        class_names: 类别名称字典
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config.get('names', {})


def generate_color(index):
    """
    根据索引生成颜色

    Args:
        index: 类别索引

    Returns:
        color: (B, G, R)颜色值
    """
    # 使用HSL颜色空间生成均匀分布的颜色
    hue = (index * 137.5) % 360  # 使用黄金角分割，生成均匀分布的颜色
    saturation = 80  # 饱和度
    lightness = 60  # 亮度

    # 转换HSL到RGB
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
    return (int(b * 255), int(g * 255), int(r * 255))


def annotate_image(image_path, annotation_path, class_names, show_label=True):
    """
    在图像上绘制真值标注

    Args:
        image_path: 图像文件路径
        annotation_path: 标注文件路径
        class_names: 类别名称字典
        show_label: 是否显示类别名称
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 读取标注文件
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    encountered_classes = set()

    # 绘制每个标注
    for annotation in annotations:
        parts = annotation.strip().split(' ')
        if len(parts) != 5:
            continue

        # 解析标注信息 - YOLO格式：class_index x_center y_center width height
        class_index, x_center, y_center, width, height = map(float, parts)

        # 将归一化的坐标转换为像素坐标
        img_height, img_width, _ = image.shape
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # 计算框的左上角和右下角
        left = int(x_center - width / 2)
        top = int(y_center - height / 2)
        right = int(x_center + width / 2)
        bottom = int(y_center + height / 2)

        # 获取类别名称和颜色
        class_name = class_names.get(int(class_index), f'unknown_{class_index}')
        color = generate_color(int(class_index))

        # 绘制边界框
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # 绘制类别标签
        if show_label:
            label = f'{class_name}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_label = max(top, label_size[1])

            # 绘制标签背景
            cv2.rectangle(image,
                          (left, top_label - label_size[1]),
                          (left + label_size[0], top_label + base_line),
                          color, cv2.FILLED)

            # 绘制标签文本
            cv2.putText(image, label, (left, top_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def process_directory(images_dir, annotations_dir, class_names, output_dir=None, show_label=True):
    """
    处理整个目录中的图像和标注

    Args:
        images_dir: 图像目录路径
        annotations_dir: 标注目录路径
        class_names: 类别名称字典
        output_dir: 输出目录路径，若为None则显示图像
        show_label: 是否在框上显示类别名称
    """
    # 如果需要保存，创建输出目录
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # 记录所有遇到的类别
    all_encountered_classes = set()

    for image_file in image_files:
        # 构建完整路径
        image_path = os.path.join(images_dir, image_file)
        annotation_file = image_file.replace('.jpg', '.txt')
        annotation_path = os.path.join(annotations_dir, annotation_file)

        # 检查标注文件是否存在
        if not os.path.exists(annotation_path):
            print(f"标注文件不存在: {annotation_path}")
            continue

        # 构建输出路径
        if output_dir is not None:
            output_path = os.path.join(output_dir, image_file)
        else:
            output_path = None

        # 处理图像
        annotated_image = annotate_image(image_path, annotation_path, class_names, show_label=show_label)

        # 保存或显示图像
        if annotated_image is not None:
            if output_dir is not None:
                cv2.imwrite(output_path, annotated_image)
                print(f"已保存标注图像: {output_path}")
            else:
                cv2.imshow('Annotated Image', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 输出所有遇到的类别统计
    print(f"\n所有图像中遇到的类别: {sorted(all_encountered_classes)}")
    print(f"已知类别: {sorted(class_names.keys())}")
    print(f"未知类别: {sorted(all_encountered_classes - set(class_names.keys()))}")


if __name__ == '__main__':
    SHOW_LABEL = True  # 控制是否显示类别名称

    # 设置路径
    images_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-test-dev/images'
    labels_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-test-dev/labels'
    output_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-test-dev/labels_images'

    # 加载VisDrone类别定义
    yaml_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml'
    class_names = load_visdrone_classes(yaml_path)

    # 确保类别字典是完整的
    print("加载的类别定义:")
    for idx, name in sorted(class_names.items()):
        print(f"  {idx}: {name}")

    # 处理整个目录
    process_directory(images_dir, labels_dir, class_names, output_dir, show_label=SHOW_LABEL)
    print("处理完成!")
