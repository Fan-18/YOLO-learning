import torch
import torch.nn as nn
import cv2
import numpy as np

def preprocess_image(img, target_size=(640, 640)):
    """
    针对 YOLO11 的预处理逻辑
    1. 缩放图片 (Letterbox)
    2. BGR 转 RGB
    3. 归一化 (0-1)
    4. 增加 Batch 维度并转为 Tensor
    """
    # 调整大小，保持长宽比（这里简化为直接缩放，YOLO 内部通常用 letterbox）
    img_resized = cv2.resize(img, target_size)
    # BGR 转 RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 归一化并将形状从 (H, W, C) 转为 (C, H, W)
    img_tensor = img_rgb.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).float()
    img_tensor /= 255.0  # 归一化
    # 增加 Batch 维度 [1, 3, 640, 640]
    return img_tensor.unsqueeze(0)


def draw_on_image(img, heatmap):
    """
    将热力图叠加到原图上
    """
    # 1. 将 0-1 的热力图转换为 0-255 的灰度图
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # 2. 将热力图缩放到原始图片大小
    heatmap_resized = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))
    
    # 3. 应用伪彩色映射 (JET 模式：蓝色为低，红色为高)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # 4. 将伪彩色热力图与原图融合
    # alpha 是原图透明度，1-alpha 是热力图透明度
    alpha = 0.6
    result = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)
    
    return result

class YOLO11GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def save_gradient(module, grad_input, grad_output):
            # 获取输出的梯度
            self.gradients = grad_output[0].detach()

        def save_activation(module, input, output):
            # 获取前向传播特征图
            self.activations = output.detach()

        self.hooks.append(self.target_layers.register_forward_hook(save_activation))
        self.hooks.append(self.target_layers.register_full_backward_hook(save_gradient))

    def generate_heatmap(self, input_tensor, class_idx=None):
        # --- 修改点 1：开启梯度计算环境 ---
        with torch.enable_grad():
            # 确保输入张量也开启梯度
            input_tensor.requires_grad_(True)
            
            # 1. 前向传播
            preds = self.model(input_tensor)
            
            # 2. 获取预测值
            output = preds[0][0] 
            cls_probs = output[4:] 

            if class_idx is None:
                # score = torch.max(cls_probs) 
                score = cls_probs[class_idx].sum()
            else:
                score = torch.max(cls_probs[class_idx])

            # --- 修改点 2：检查 score 是否可以求导 ---
            if score.grad_fn is None:
                raise RuntimeError("Score tensor has no grad_fn. Ensure model is in a state that allows gradients.")

            # 3. 反向传播
            self.model.zero_grad()
            score.backward(retain_graph=True)

            if self.gradients is None:
                return None

            # 计算通道权重
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            # 加权求和
            cam = torch.sum(weights * self.activations, dim=1).squeeze()
            
            # 处理结果
            cam = np.maximum(cam.cpu().numpy(), 0)
            # 针对遥感图像的细节，建议使用更高的插值质量
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]), interpolation=cv2.INTER_LANCZOS4)
            
            # 归一化
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
                
            return cam

    def remove_hooks(self):
            """
            循环结束后必须调用，以清除模型中的 Hook，防止影响后续计算
            """
            for hook in self.hooks:
                hook.remove()
            self.hooks = [] # 清空列表

