import os
import cv2
import argparse
import numpy as np
import onnxruntime
from tqdm import tqdm

def natural_sort_key(s):
    """提供自然排序的键函数"""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def load_image(path):
    """加载并预处理图像"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Error loading image: {path}")
    
    # 转换为 RGB 并归一化到 [0,1]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # 转换为 NCHW 格式
    if len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # CHW -> NCHW
    return img


def RGB2YCrCb(rgb):
    """RGB 转 YCrCb"""
    # 确保输入是 NCHW 格式
    assert len(rgb.shape) == 4 and rgb.shape[1] == 3
    
    # 转换矩阵
    transform = np.array([
        [0.299, 0.587, 0.114],      # Y
        [0.5, -0.4187, -0.0813],    # Cr
        [-0.1687, -0.3313, 0.5]     # Cb
    ]).astype(np.float32)
    
    # 重塑数据以进行矩阵乘法
    n, c, h, w = rgb.shape
    rgb_reshaped = rgb.transpose(0, 2, 3, 1).reshape(-1, 3)
    
    # 进行颜色空间转换
    ycrcb = np.dot(rgb_reshaped, transform.T)
    
    # 重塑回原始维度
    ycrcb = ycrcb.reshape(n, h, w, 3).transpose(0, 3, 1, 2)
    
    # 分离通道
    return (
        ycrcb[:, 0:1, :, :],  # Y
        ycrcb[:, 1:2, :, :],  # Cr
        ycrcb[:, 2:3, :, :]   # Cb
    )

def YCrCb2RGB(y, cr, cb):
    """YCrCb 转 RGB"""
    # 合并通道
    ycrcb = np.concatenate([y, cr, cb], axis=1)
    
    # 转换矩阵
    transform = np.array([
        [1.0, 1.403, 0.0],
        [1.0, -0.714, -0.344],
        [1.0, 0.0, 1.773]
    ]).astype(np.float32)
    
    # 重塑数据
    n, c, h, w = ycrcb.shape
    ycrcb_reshaped = ycrcb.transpose(0, 2, 3, 1).reshape(-1, 3)
    
    # 进行颜色空间转换
    rgb = np.dot(ycrcb_reshaped, transform.T)
    
    # 重塑回原始维度
    rgb = rgb.reshape(n, h, w, 3).transpose(0, 3, 1, 2)
    
    # 裁剪到 [0,1]
    return np.clip(rgb, 0, 1)

def save_image(img, path):
    """保存图像"""
    # 转换为 HWC 格式
    img = np.transpose(img[0], (1, 2, 0))
    
    # 转换为 BGR 并缩放到 [0,255]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def main(ir_dir, vi_dir, save_dir, fusion_model_path):
    # 初始化 ONNX Runtime (只使用 CPU)
    ort_session = onnxruntime.InferenceSession(
        fusion_model_path,
        providers=['CPUExecutionProvider']
    )# 打印模型输入输出信息
    print("Model inputs:", [input.name for input in ort_session.get_inputs()])
    print("Model outputs:", [output.name for output in ort_session.get_outputs()])
    print('ONNX model loaded successfully!')

    # 获取所有图像文件并自然排序
    vi_files = sorted([f for f in os.listdir(vi_dir) if f.endswith(('.png', '.jpg', '.bmp'))],
                     key=natural_sort_key)
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.bmp'))],
                     key=natural_sort_key)

    if len(vi_files) != len(ir_files):
        raise ValueError("Number of visible and infrared images don't match!")

    # 处理每对图像
    for vi_name, ir_name in tqdm(zip(vi_files, ir_files), total=len(vi_files)):
        # 加载图像
        img_vis = load_image(os.path.join(vi_dir, vi_name))
        img_ir = load_image(os.path.join(ir_dir, ir_name))
        
        # 如果红外图像是 3 通道，转换为单通道
        if img_ir.shape[1] == 3:
            img_ir = np.mean(img_ir, axis=1, keepdims=True)

        # RGB to YCrCb
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)

        # ONNX 推理 - 使用正确的输入名称
        ort_inputs = {
            'l_image_vis_': vi_Y.astype(np.float32),
            'l_image_ir_': img_ir.astype(np.float32)
        }
        fused_img = ort_session.run(None, ort_inputs)[0]

        # YCrCb to RGB
        fused_img = YCrCb2RGB(fused_img, vi_Cb, vi_Cr)

        # 保存结果
        save_path = os.path.join(save_dir, f"fused_{vi_name}")
        save_image(fused_img, save_path)
        print(f'Processed: {vi_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusion with ONNX')
    parser.add_argument('--model_path', '-M', type=str, 
                       default='./model/Fusion/fusion_model.onnx')
    parser.add_argument('--ir_dir', '-ir_dir', type=str, 
                       default='./test_imgs/ir')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, 
                       default='./test_imgs/vi')
    parser.add_argument('--save_dir', '-save_dir', type=str, 
                       default='./SeAFusion')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('Testing SeAFusion with ONNX Runtime')
    
    main(args.ir_dir, args.vi_dir, args.save_dir, args.model_path)