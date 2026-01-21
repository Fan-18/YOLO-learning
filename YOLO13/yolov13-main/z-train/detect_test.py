import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
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


def annotate_ground_truth(image_path, annotation_path, class_names, show_label=True):
    """
    在图像上绘制真值标注
    show_label: 是否在框上显示类别名称
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    encountered_classes = set()

    for annotation in annotations:
        parts = annotation.strip().split(' ')
        if len(parts) != 5:
            continue

        # YOLO格式：class_index x_center y_center width height
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

        class_name = class_names.get(int(class_index), f'unknown_{class_index}')
        color = generate_color(int(class_index))

        cv2.rectangle(image, (left, top), (right, bottom), color, 1)

        # ✅ 只有开关打开才画类别名
        if show_label:
            label = f'{class_name}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_label = max(top, label_size[1])

            cv2.rectangle(
                image,
                (left, top_label - label_size[1]),
                (left + label_size[0], top_label + base_line),
                color, cv2.FILLED
            )
            cv2.putText(image, label, (left, top_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print(f"图像 {os.path.basename(image_path)} 中遇到的类别: {sorted(encountered_classes)}")
    return image


def annotate_prediction(image_path, model, class_names, show_label=True, show_conf=True):
    """
    使用YOLO模型进行推理并在图像上绘制预测结果
    show_label: 是否在框上显示类别名称
    show_conf: 是否显示置信度（可选，顺手也给你留了）
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    results = model(image)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            class_name = class_names.get(cls, f'unknown_{cls}')
            color = generate_color(cls)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

            # ✅ 只有开关打开才画类别名/置信度
            if show_label:
                if show_conf:
                    label = f'{class_name} {conf:.2f}'
                else:
                    label = f'{class_name}'

                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top_label = max(y1, label_size[1])

                cv2.rectangle(
                    image,
                    (x1, top_label - label_size[1]),
                    (x1 + label_size[0], top_label + base_line),
                    color, cv2.FILLED
                )
                cv2.putText(image, label, (x1, top_label),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def process_image(image_path, annotation_path, model, class_names, output_path=None, show_label=True):
    """
    处理单张图像，绘制真值和预测结果并排显示
    show_label: 是否在框上显示类别名称
    """
    gt_image = annotate_ground_truth(image_path, annotation_path, class_names, show_label=show_label)
    if gt_image is None:
        return

    pred_image = annotate_prediction(image_path, model, class_names, show_label=show_label)
    if pred_image is None:
        return

    height = gt_image.shape[0]
    width = gt_image.shape[1]
    combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
    combined[:, :width] = gt_image
    combined[:, width:] = pred_image

    cv2.putText(combined, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'Prediction', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if output_path is not None:
        cv2.imwrite(output_path, combined)
        print(f"已保存组合图像: {output_path}")
    else:
        cv2.imshow('Ground Truth vs Prediction', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_directory(images_dir, annotations_dir, model, class_names, output_dir=None, show_label=True):
    """
    处理整个目录中的图像和标注
    show_label: 是否在框上显示类别名称
    """
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    all_encountered_classes = set()

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        annotation_file = image_file.replace('.jpg', '.txt')
        annotation_path = os.path.join(annotations_dir, annotation_file)

        if not os.path.exists(annotation_path):
            print(f"标注文件不存在: {annotation_path}")
            continue

        if output_dir is not None:
            output_path = os.path.join(output_dir, image_file)
        else:
            output_path = None

        process_image(
            image_path, annotation_path, model, class_names,
            output_path=output_path,
            show_label=show_label
        )

    print(f"\n所有图像中遇到的类别: {sorted(all_encountered_classes)}")
    print(f"已知类别: {sorted(class_names.keys())}")
    print(f"未知类别: {sorted(all_encountered_classes - set(class_names.keys()))}")



if __name__ == '__main__':
    SHOW_LABEL = False  # 控制是否显示类别名称

    # 设置路径
    # images_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-test-dev/images'
    # labels_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-test-dev/labels'
    # # output_dir = '/home/jack/11/1027/YOLO13/yolov13-main/runs/yolo11x-RS.yaml_VisDrone-300/test_images_nolabel'
    # output_dir = '/home/jack/11/1027/YOLO13/yolov13-main/runs/yolo11x-RS.yaml_VisDrone-300/test_images'
    # VISO-car
    images_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VISO/Detection_yolo_format/car/test/images'
    labels_dir = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VISO/Detection_yolo_format/car/test/labels'
    # output_dir = '/home/jack/11/1027/YOLO13/yolov13-main/runs/yolo11x-RS.yaml_VisDrone-300/test_images_nolabel'
    output_dir = '/home/jack/11/1027/YOLO13/yolov13-main/runs/VISO_Detection/car/1_yolo11n_VISO_1024/test_images'



    # 加载类别定义
    yaml_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VISO_Detection.yaml'
    class_names = load_visdrone_classes(yaml_path)

    # 加载YOLO模型
    model_path = '/home/jack/11/1027/YOLO13/yolov13-main/runs/VISO_Detection/car/1_yolo11n_VISO_1024/weights/best.pt'
    model = YOLO(model_path)

    # 处理整个目录
    process_directory(images_dir, labels_dir, model, class_names, output_dir, show_label=SHOW_LABEL)
    print("处理完成!")
